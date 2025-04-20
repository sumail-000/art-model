import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

# Import core components
from multimodal.encoder import CLIPLikeEncoder
from multimodal.intent_parser import DynamicIntentParser
from generative.pixel_gan import HierarchicalPixelGAN, ScaffoldNetwork
from datasets.pixel_art_dataset import PixelArtDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pixelmind-x-training")


class Trainer:
    """Main trainer class for PixelMind-X"""
    
    def __init__(self, config_path):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir="runs/pixelmind_training")
        
        # Initialize components
        self._init_datasets()
        self._init_models()
        self._init_optimizers()
        
    def _init_datasets(self):
        """Initialize datasets and dataloaders"""
        logger.info("Initializing datasets")
        
        # Get dataset config
        dataset_config = self.config["data"]["dataset"]
        loader_config = self.config["data"]["data_loaders"]
        aug_config = self.config["data"]["augmentation"]
        
        # Initialize training dataset
        self.train_dataset = PixelArtDataset(
            sources=dataset_config["sources"],
            split="train",
            resolution=dataset_config["image_resolution"],
            augmentation=aug_config["enabled"]
        )
        
        # Initialize validation dataset
        self.val_dataset = PixelArtDataset(
            sources=dataset_config["sources"],
            split="val",
            resolution=dataset_config["image_resolution"],
            augmentation=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=True,
            num_workers=loader_config["num_workers"],
            pin_memory=loader_config["pin_memory"],
            prefetch_factor=loader_config["prefetch_factor"]
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=loader_config["num_workers"],
            pin_memory=loader_config["pin_memory"]
        )
        
        logger.info(f"Initialized datasets - Training: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}")
    
    def _init_models(self):
        """Initialize model components"""
        logger.info("Initializing models")
        
        # Initialize CLIP-like encoder
        encoder_config = self.config["multimodal_understanding"]["encoder"]
        self.encoder = CLIPLikeEncoder(
            embed_dim=encoder_config["embedding_dim"],
            text_model_name=encoder_config["text_model"],
            vision_model_name=encoder_config["vision_model"],
            audio_model_name=encoder_config["audio_model"]
        ).to(self.device)
        
        # Initialize PixelGAN
        gan_config = self.config["generative_core"]["hierarchical_pixelgan"]
        self.pixel_gan = HierarchicalPixelGAN(
            latent_dim=gan_config["latent_dim"],
            condition_dim=gan_config["condition_dim"]
        ).to(self.device)
        
        # Initialize intent parser
        intent_config = self.config["multimodal_understanding"]["intent_parser"]
        self.intent_parser = DynamicIntentParser(
            base_model_name=intent_config["base_model"],
            max_input_length=intent_config["max_input_length"],
            max_output_length=intent_config["max_output_length"],
            embedding_dim=encoder_config["embedding_dim"]
        ).to(self.device)
        
        logger.info("Models initialized")
    
    def _init_optimizers(self):
        """Initialize optimizers for each model component"""
        logger.info("Initializing optimizers")
        
        # Get optimizer configs
        clip_config = self.config["pre_training"]["clip_encoder"]
        gan_config = self.config["pre_training"]["pixel_gan"]
        intent_config = self.config["pre_training"]["intent_parser"]
        
        # CLIP-like encoder optimizer
        self.encoder_optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=clip_config["learning_rate"],
            weight_decay=clip_config["weight_decay"]
        )
        
        # PixelGAN optimizer
        self.pixel_gan_optimizer = optim.AdamW(
            self.pixel_gan.parameters(),
            lr=gan_config["learning_rate"]
        )
        
        # Intent parser optimizer
        self.intent_parser_optimizer = optim.AdamW(
            self.intent_parser.parameters(),
            lr=intent_config["learning_rate"],
            weight_decay=intent_config["weight_decay"]
        )
        
        logger.info("Optimizers initialized")
    
    def train_encoder(self, num_epochs=None):
        """Train the CLIP-like encoder"""
        logger.info("Starting encoder training")
        
        # Get clip config
        clip_config = self.config["pre_training"]["clip_encoder"]
        epochs = num_epochs if num_epochs is not None else clip_config["epochs"]
        temperature = clip_config["temperature"]
        
        # Define loss function
        def contrastive_loss(image_features, text_features, temperature=0.07):
            # Normalized features
            image_features = nn.functional.normalize(image_features, dim=1)
            text_features = nn.functional.normalize(text_features, dim=1)
            
            # Cosine similarity as logits
            logits = (text_features @ image_features.T) / temperature
            
            # Labels (diagonal is the positive pair)
            labels = torch.arange(len(image_features), device=self.device)
            
            # Calculate loss
            loss_i = nn.functional.cross_entropy(logits, labels)
            loss_t = nn.functional.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            return loss
        
        # Training loop
        global_step = 0
        for epoch in range(epochs):
            self.encoder.train()
            epoch_loss = 0.0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    # Move data to device
                    text_input_ids = batch["text_input_ids"].to(self.device)
                    text_attention_mask = batch["text_attention_mask"].to(self.device)
                    pixel_values = batch["pixel_values"].to(self.device)
                    
                    # Forward pass
                    text_features = self.encoder.encode_text(text_input_ids, text_attention_mask)
                    image_features = self.encoder.encode_image(pixel_values)
                    
                    # Calculate loss
                    loss = contrastive_loss(image_features, text_features, temperature)
                    
                    # Backward and optimize
                    self.encoder_optimizer.zero_grad()
                    loss.backward()
                    self.encoder_optimizer.step()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Log to tensorboard
                    self.writer.add_scalar("Encoder/Train/Loss", loss.item(), global_step)
                    global_step += 1
            
            # Validation step
            self.encoder.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    # Move data to device
                    text_input_ids = batch["text_input_ids"].to(self.device)
                    text_attention_mask = batch["text_attention_mask"].to(self.device)
                    pixel_values = batch["pixel_values"].to(self.device)
                    
                    # Forward pass
                    text_features = self.encoder.encode_text(text_input_ids, text_attention_mask)
                    image_features = self.encoder.encode_image(pixel_values)
                    
                    # Calculate loss
                    loss = contrastive_loss(image_features, text_features, temperature)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            self.writer.add_scalar("Encoder/Val/Loss", val_loss, epoch)
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint("encoder", epoch)
        
        logger.info("Encoder training completed")
    
    def train_pixel_gan(self, resolution=16, num_epochs=None):
        """Train the PixelGAN at a specific resolution"""
        logger.info(f"Starting PixelGAN training at resolution {resolution}")
        
        # Get gan config
        gan_config = self.config["pre_training"]["pixel_gan"]
        epochs = num_epochs if num_epochs is not None else gan_config["epochs_per_resolution"][str(resolution)]
        batch_size = gan_config["batch_sizes"][str(resolution)]
        
        # Train the appropriate resolution component
        if resolution == 16:
            # Train scaffold network
            model = self.pixel_gan.scaffold_net
            # Update dataloader batch size
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config["data"]["data_loaders"]["num_workers"],
                pin_memory=self.config["data"]["data_loaders"]["pin_memory"]
            )
            
        elif resolution == 64:
            # Train mid-resolution network
            model = self.pixel_gan.refinement_net.mid_unet
            
        elif resolution == 256:
            # Train high-resolution network
            model = self.pixel_gan.refinement_net.high_unet
        
        # Training loop
        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    # Move data to device
                    pixel_values = batch["pixel_values"].to(self.device)
                    text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
                    
                    # If we don't have text embeddings, generate them
                    if text_embeddings is None and "text_input_ids" in batch:
                        with torch.no_grad():
                            text_input_ids = batch["text_input_ids"].to(self.device)
                            text_attention_mask = batch["text_attention_mask"].to(self.device)
                            text_embeddings = self.encoder.encode_text(text_input_ids, text_attention_mask)
                    
                    # Forward pass
                    # This is a simplified version; in reality, the GAN training would be more complex
                    # with proper noise scheduling and diffusion steps
                    noise = torch.randn_like(pixel_values)
                    timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=self.device)
                    
                    # For scaffold network training
                    if resolution == 16:
                        latent = torch.randn(pixel_values.shape[0], self.pixel_gan.scaffold_net.latent_dim, device=self.device)
                        noise_pred = model(latent, text_embeddings, timesteps)
                        loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    # Backward and optimize
                    self.pixel_gan_optimizer.zero_grad()
                    loss.backward()
                    self.pixel_gan_optimizer.step()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Log to tensorboard
                    self.writer.add_scalar(f"PixelGAN_{resolution}/Train/Loss", loss.item(), global_step)
                    global_step += 1
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - PixelGAN {resolution} Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f"pixel_gan_{resolution}", epoch)
        
        logger.info(f"PixelGAN training at resolution {resolution} completed")
    
    def train_intent_parser(self, num_epochs=None):
        """Train the intent parser"""
        logger.info("Starting intent parser training")
        
        # Get intent config
        intent_config = self.config["pre_training"]["intent_parser"]
        epochs = num_epochs if num_epochs is not None else intent_config["epochs"]
        
        # Training loop
        global_step = 0
        for epoch in range(epochs):
            self.intent_parser.train()
            epoch_loss = 0.0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    # Move data to device
                    text_input_ids = batch["text_input_ids"].to(self.device)
                    text_attention_mask = batch["text_attention_mask"].to(self.device)
                    target_intent_ids = batch["target_intent_ids"].to(self.device) if "target_intent_ids" in batch else None
                    
                    # Skip if no target intents (for datasets without intent annotations)
                    if target_intent_ids is None:
                        continue
                    
                    # Forward pass through the intent parser
                    outputs = self.intent_parser.base_model(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                        labels=target_intent_ids
                    )
                    
                    loss = outputs.loss
                    
                    # Backward and optimize
                    self.intent_parser_optimizer.zero_grad()
                    loss.backward()
                    self.intent_parser_optimizer.step()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Log to tensorboard
                    self.writer.add_scalar("IntentParser/Train/Loss", loss.item(), global_step)
                    global_step += 1
            
            # Validation step
            self.intent_parser.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    # Move data to device
                    text_input_ids = batch["text_input_ids"].to(self.device)
                    text_attention_mask = batch["text_attention_mask"].to(self.device)
                    target_intent_ids = batch["target_intent_ids"].to(self.device) if "target_intent_ids" in batch else None
                    
                    # Skip if no target intents
                    if target_intent_ids is None:
                        continue
                    
                    # Forward pass
                    outputs = self.intent_parser.base_model(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                        labels=target_intent_ids
                    )
                    
                    val_loss += outputs.loss.item()
                    
            if val_loss > 0:
                val_loss /= len(self.val_loader)
                self.writer.add_scalar("IntentParser/Val/Loss", val_loss, epoch)
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Intent Parser Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint("intent_parser", epoch)
        
        logger.info("Intent parser training completed")
    
    def save_checkpoint(self, model_type, epoch):
        """Save a checkpoint for the specified model"""
        os.makedirs("checkpoints", exist_ok=True)
        
        if model_type == "encoder":
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.encoder.state_dict(),
                "optimizer_state_dict": self.encoder_optimizer.state_dict()
            }, f"checkpoints/encoder_epoch_{epoch}.pt")
            
        elif "pixel_gan" in model_type:
            resolution = model_type.split("_")[-1]
            if resolution == "16":
                model = self.pixel_gan.scaffold_net
            elif resolution == "64":
                model = self.pixel_gan.refinement_net.mid_unet
            elif resolution == "256":
                model = self.pixel_gan.refinement_net.high_unet
            else:
                model = self.pixel_gan
                
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.pixel_gan_optimizer.state_dict()
            }, f"checkpoints/{model_type}_epoch_{epoch}.pt")
            
        elif model_type == "intent_parser":
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.intent_parser.state_dict(),
                "optimizer_state_dict": self.intent_parser_optimizer.state_dict()
            }, f"checkpoints/intent_parser_epoch_{epoch}.pt")
    
    def load_checkpoint(self, model_type, epoch):
        """Load a checkpoint for the specified model"""
        checkpoint_path = f"checkpoints/{model_type}_epoch_{epoch}.pt"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            if model_type == "encoder":
                self.encoder.load_state_dict(checkpoint["model_state_dict"])
                self.encoder_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
            elif "pixel_gan" in model_type:
                resolution = model_type.split("_")[-1]
                if resolution == "16":
                    model = self.pixel_gan.scaffold_net
                elif resolution == "64":
                    model = self.pixel_gan.refinement_net.mid_unet
                elif resolution == "256":
                    model = self.pixel_gan.refinement_net.high_unet
                else:
                    model = self.pixel_gan
                    
                model.load_state_dict(checkpoint["model_state_dict"])
                self.pixel_gan_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
            elif model_type == "intent_parser":
                self.intent_parser.load_state_dict(checkpoint["model_state_dict"])
                self.intent_parser_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint["epoch"]
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
    
    def run_pre_training(self):
        """Run the full pre-training process"""
        logger.info("Starting pre-training process")
        
        # 1. Train the encoder
        self.train_encoder()
        
        # 2. Train the PixelGAN at different resolutions
        self.train_pixel_gan(resolution=16)  # Scaffold network
        self.train_pixel_gan(resolution=64)  # Mid-resolution
        self.train_pixel_gan(resolution=256)  # High-resolution
        
        # 3. Train the intent parser
        self.train_intent_parser()
        
        logger.info("Pre-training completed")


def main():
    parser = argparse.ArgumentParser(description="PixelMind-X Training")
    parser.add_argument("--config", type=str, default="configs/training_config.json", help="Path to training config")
    parser.add_argument("--model", type=str, choices=["encoder", "pixel_gan_16", "pixel_gan_64", "pixel_gan_256", "intent_parser", "all"], default="all", help="Model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--load_checkpoint", type=int, default=None, help="Epoch to load from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Load checkpoint if specified
    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.model, args.load_checkpoint)
    
    # Run training based on model selection
    if args.model == "encoder" or args.model == "all":
        trainer.train_encoder(args.epochs)
        
    if args.model == "pixel_gan_16" or args.model == "all":
        trainer.train_pixel_gan(resolution=16, num_epochs=args.epochs)
        
    if args.model == "pixel_gan_64" or args.model == "all":
        trainer.train_pixel_gan(resolution=64, num_epochs=args.epochs)
        
    if args.model == "pixel_gan_256" or args.model == "all":
        trainer.train_pixel_gan(resolution=256, num_epochs=args.epochs)
        
    if args.model == "intent_parser" or args.model == "all":
        trainer.train_intent_parser(args.epochs)
        
    if args.model == "all":
        trainer.run_pre_training()


if __name__ == "__main__":
    main() 