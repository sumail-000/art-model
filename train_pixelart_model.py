import os
import json
import argparse
import torch
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
from accelerate import Accelerator
from torchvision import transforms

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Set random seed
    set_seed(args.seed)
    
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
        num_workers=args.dataloader_num_workers
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
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            print("bitsandbytes not found, using regular AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    # Prepare for training
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader
    )
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # Get total number of training steps
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    
    # Start training
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
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
                    ).to(accelerator.device)
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
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log training info
                if global_step % args.log_interval == 0:
                    accelerator.print(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.detach().item()}")
                
                # Save checkpoint
                if global_step % args.save_interval == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Unwrap model
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        
                        # Save UNet model
                        unwrapped_unet.save_pretrained(os.path.join(save_dir, "unet"))
                        
                        # Save as full Stable Diffusion pipeline
                        pipeline = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unwrapped_unet,
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            requires_safety_checker=False
                        )
                        pipeline.save_pretrained(save_dir)
                        
                        # Save training args
                        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
                            json.dump(vars(args), f, indent=2)
                
                # Generate sample images
                if args.validation_prompts and global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        with torch.autocast("cuda", enabled=accelerator.mixed_precision == "fp16"):
                            validate_and_generate_samples(
                                accelerator, vae, text_encoder, tokenizer, 
                                unet, args, global_step, noise_scheduler
                            )
            
            # Check if we've reached max steps
            if global_step >= args.max_train_steps:
                break
    
    # Final save
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_dir, exist_ok=True)
        
        # Unwrap model
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Save UNet model
        unwrapped_unet.save_pretrained(os.path.join(save_dir, "unet"))
        
        # Save as full Stable Diffusion pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unwrapped_unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.save_pretrained(save_dir)
        
        # Save training args
        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    accelerator.end_training()

def validate_and_generate_samples(accelerator, vae, text_encoder, tokenizer, unet, args, step, noise_scheduler):
    """Generate sample images during training to monitor progress"""
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline = pipeline.to(accelerator.device)
    
    # Create output directory
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Set to eval mode
    pipeline.unet.eval()
    
    # Process each validation prompt
    for i, prompt in enumerate(args.validation_prompts):
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        images = pipeline(
            prompt, 
            num_inference_steps=args.validation_steps_num,
            guidance_scale=args.validation_guidance_scale,
            generator=generator
        ).images
        
        # Save the image
        for j, image in enumerate(images):
            image.save(os.path.join(samples_dir, f"step_{step}_prompt_{i}_sample_{j}.png"))
    
    # Set back to training mode
    pipeline.unet.train()
    del pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PixelArt model")
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="Learning rate scheduler (constant, linear, cosine, cosine_with_restarts)")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"], 
                        help="Mixed precision training")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Checkpoint saving interval")
    parser.add_argument("--validation_prompts", nargs="+", type=str,
                        default=["pixel art landscape", "pixel art character", "pixel art building", "pixel art game scene"],
                        help="Prompts to use for validation")
    parser.add_argument("--validation_steps", type=int, default=500,
                        help="Steps between validations")
    parser.add_argument("--validation_steps_num", type=int, default=30,
                        help="Number of denoising steps for validation")
    parser.add_argument("--validation_guidance_scale", type=float, default=7.5,
                        help="Guidance scale for validation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args) 