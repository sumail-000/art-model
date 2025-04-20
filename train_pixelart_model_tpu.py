import os
import json
import argparse
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDPMScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchvision import transforms

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Dataset class for loading pixel art images and prompts
class PixelArtDataset(Dataset):
    def __init__(self, dataset_dir, split="train", transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_file = os.path.join(dataset_dir, f"{split}_metadata.json")
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} {split} examples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image_path = os.path.join(self.dataset_dir, item["image_path"])
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get the caption
        caption = item["description"]
        
        return {
            "image": image,
            "caption": caption
        }

def train(args):
    # Set random seed
    set_seed(args.seed)
    
    # Set up TPU device
    device = xm.xla_device()
    
    # Set up transformations
    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create train and validation datasets
    train_dataset = PixelArtDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        transform=train_transforms
    )
    
    val_dataset = PixelArtDataset(
        dataset_dir=args.dataset_dir,
        split="val",
        transform=train_transforms
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet"
    )
    
    # Move models to TPU
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    # Freeze vae and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Move to device and set to eval mode
    vae.eval()
    text_encoder.eval()
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    # Prepare dataloaders for TPU
    train_loader = pl.MpDeviceLoader(train_dataloader, device)
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # Training loop
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")
    global_step = 0
    
    # Start training
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_loader):
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["image"]).latent_dist.sample() * 0.18215
            
            # Process text
            with torch.no_grad():
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # Add noise to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            
            # Back-propagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # TPU sync
            xm.mark_step()
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Log training info
            if global_step % args.log_interval == 0:
                loss_value = loss.detach().item()
                print(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss_value}")
                xm.master_print(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss_value}")
            
            # Save checkpoint
            if global_step % args.save_interval == 0 and global_step > 0:
                if xm.is_master_ordinal():
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Save UNet model - need to move to CPU first
                    unet_cpu = UNet2DConditionModel.from_pretrained(
                        args.pretrained_model_name_or_path, 
                        subfolder="unet"
                    )
                    unet_cpu.load_state_dict({k: v.cpu() for k, v in unet.state_dict().items()})
                    unet_cpu.save_pretrained(os.path.join(save_dir, "unet"))
                    del unet_cpu
                    
                    # Save training args
                    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
                        json.dump(vars(args), f, indent=2)
                    
                    print(f"Saved checkpoint at step {global_step}")
            
            # Check if we've reached max steps
            if global_step >= args.max_train_steps:
                break
    
    # Final save
    if xm.is_master_ordinal():
        save_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save UNet model - need to move to CPU first
        unet_cpu = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="unet"
        )
        unet_cpu.load_state_dict({k: v.cpu() for k, v in unet.state_dict().items()})
        unet_cpu.save_pretrained(os.path.join(save_dir, "unet"))
        
        # Save other models to create a full pipeline
        tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        text_encoder_cpu = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )
        text_encoder_cpu.save_pretrained(os.path.join(save_dir, "text_encoder"))
        
        vae_cpu = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae"
        )
        vae_cpu.save_pretrained(os.path.join(save_dir, "vae"))
        
        noise_scheduler.save_pretrained(os.path.join(save_dir, "scheduler"))
        
        # Save training args
        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    xm.master_print("Training complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PixelArt model on TPU")
    parser.add_argument("--dataset_dir", type=str, default="data/pixel_art_dataset",
                        help="Path to the pixel art dataset directory")
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained Stable Diffusion model")
    parser.add_argument("--output_dir", type=str, default="output/pixelart_model",
                        help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution for training")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--max_train_steps", type=int, default=10000,
                        help="Total number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="Learning rate scheduler (constant, linear, cosine, cosine_with_restarts)")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint saving interval")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args) 