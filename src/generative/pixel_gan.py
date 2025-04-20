import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import math

class AttentionGate(nn.Module):
    """
    Attention gate for focusing on important features
    """
    def __init__(self, in_channels, gate_channels, intermediate_channels):
        super().__init__()
        
        # Feature transformer for input features
        self.feat_transform = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1)
        
        # Gate transformer for gate signal
        self.gate_transform = nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1)
        
        # Attention mapper
        self.attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, gate_signal):
        # Transform input features
        feat = self.feat_transform(x)
        
        # Transform gate signal
        gate = self.gate_transform(gate_signal)
        
        # Element-wise addition followed by attention computation
        attention_map = self.attention(feat + gate)
        
        # Apply attention map to input features
        return x * attention_map
        
class ScaffoldNetwork(nn.Module):
    """
    Low-resolution scaffold network (16x16) for generating basic structure
    """
    def __init__(self, latent_dim=1024, condition_dim=1024):
        super().__init__()
        
        # Base UNet from diffusers
        self.unet = UNet2DConditionModel(
            sample_size=16,  # Low resolution 16x16
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256),
            cross_attention_dim=condition_dim,
            attention_head_dim=8,
            layers_per_block=2
        )
        
        # VAE for encoding/decoding scaffold latents 
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            block_out_channels=[128],
            latent_channels=4
        )
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012
        )
        
        # Initial projection for latent vector to spatial features
        self.latent_projector = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 16 * 16 * 4)  # 4 = latent channels
        )
        
    def encode_image(self, pixel_values):
        """Encode an image to VAE latent space"""
        return self.vae.encode(pixel_values).latent_dist.sample()
        
    def decode_latents(self, latents):
        """Decode VAE latents to image"""
        return self.vae.decode(latents).sample
    
    def forward(self, latent, condition, timesteps=None, return_dict=True):
        """
        Generate low-resolution scaffold
        
        Args:
            latent: Input latent vector (B, latent_dim)
            condition: Conditioning information (B, condition_dim)
            timesteps: Optional timesteps for diffusion scheduling
        """
        batch_size = latent.shape[0]
        
        # Project latent to spatial features
        spatial_latent = self.latent_projector(latent)
        spatial_latent = spatial_latent.reshape(batch_size, 4, 16, 16)
        
        # Get timesteps if not provided (for inference)
        if timesteps is None:
            timesteps = torch.randint(
                0, 
                self.scheduler.num_train_timesteps, 
                (batch_size,), 
                device=latent.device
            )
        
        # Forward pass through UNet
        noise_pred = self.unet(
            spatial_latent,
            timesteps,
            encoder_hidden_states=condition,
            return_dict=return_dict
        ).sample
        
        return noise_pred
    
    def generate(self, latent, condition, num_inference_steps=50):
        """Generate low-resolution scaffold through diffusion process"""
        batch_size = latent.shape[0]
        
        # Project latent to spatial features and add noise
        spatial_latent = self.latent_projector(latent)
        spatial_latent = spatial_latent.reshape(batch_size, 4, 16, 16)
        
        # Add noise
        noise = torch.randn_like(spatial_latent)
        noisy_latent = self.scheduler.add_noise(
            spatial_latent, 
            noise, 
            torch.tensor([self.scheduler.num_train_timesteps - 1])
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Iterative denoising
        latents = noisy_latent
        for t in self.scheduler.timesteps:
            # Get model prediction
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=condition
                ).sample
            
            # Scheduler step
            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
        
        # Decode to image
        images = self.decode_latents(latents)
        
        return {
            "images": images,
            "latents": latents
        }

class RefinementNetwork(nn.Module):
    """
    High-resolution refinement network with attention gates,
    scaling from 64x64 to 256x256
    """
    def __init__(self, 
                 low_res_size=16,
                 mid_res_size=64, 
                 high_res_size=256, 
                 latent_dim=1024,
                 condition_dim=1024):
        super().__init__()
        
        # Base UNet from diffusers for mid-resolution (64x64)
        self.mid_unet = UNet2DConditionModel(
            sample_size=mid_res_size,
            in_channels=8,  # 4 from low-res + 4 for mid-res
            out_channels=4,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 512),
            cross_attention_dim=condition_dim,
            attention_head_dim=8
        )
        
        # High-resolution UNet (256x256)
        self.high_unet = UNet2DConditionModel(
            sample_size=high_res_size,
            in_channels=8,  # 4 from mid-res + 4 for high-res
            out_channels=4,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 512, 512),
            cross_attention_dim=condition_dim,
            attention_head_dim=8
        )
        
        # Attention gates
        self.mid_attention_gate = AttentionGate(4, 4, 64)
        self.high_attention_gate = AttentionGate(4, 4, 64)
        
        # VAE for final decoding
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512],
            latent_channels=4
        )
        
        # Noise schedulers
        self.mid_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012
        )
        
        self.high_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012
        )
        
        # Low-res to mid-res upsampler
        self.low_to_mid_upsampler = nn.Sequential(
            nn.Upsample(scale_factor=mid_res_size // low_res_size, mode='bilinear'),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Mid-res to high-res upsampler
        self.mid_to_high_upsampler = nn.Sequential(
            nn.Upsample(scale_factor=high_res_size // mid_res_size, mode='bilinear'),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Latent projectors for different resolutions
        self.mid_latent_projector = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, mid_res_size * mid_res_size * 4)
        )
        
        self.high_latent_projector = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, high_res_size * high_res_size // 16 * 4)  # Projected to save memory
        )
        
    def generate(self, low_res_latents, latent, condition, num_inference_steps=50):
        """
        Generate high-resolution image from low-resolution scaffold
        
        Args:
            low_res_latents: Latents from scaffold network (B, 4, 16, 16)
            latent: Input latent vector (B, latent_dim)
            condition: Conditioning information (B, condition_dim)
            num_inference_steps: Number of diffusion steps
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # ----------- Mid-resolution generation -----------
        # Upsample low-res latents to mid-res
        upsampled_low_res = self.low_to_mid_upsampler(low_res_latents)
        
        # Project latent to mid-res spatial features
        mid_spatial_latent = self.mid_latent_projector(latent)
        mid_spatial_latent = mid_spatial_latent.reshape(batch_size, 4, 64, 64)
        
        # Add noise
        mid_noise = torch.randn_like(mid_spatial_latent)
        mid_noisy_latent = self.mid_scheduler.add_noise(
            mid_spatial_latent,
            mid_noise,
            torch.tensor([self.mid_scheduler.num_train_timesteps - 1], device=device)
        )
        
        # Apply attention gate to upsampled low-res features
        attended_low_res = self.mid_attention_gate(upsampled_low_res, low_res_latents)
        
        # Concatenate attended low-res with mid-res latents
        mid_latents = torch.cat([attended_low_res, mid_noisy_latent], dim=1)
        
        # Set up scheduler
        self.mid_scheduler.set_timesteps(num_inference_steps)
        
        # Iterative denoising for mid-res
        for t in self.mid_scheduler.timesteps:
            # Get model prediction
            with torch.no_grad():
                noise_pred = self.mid_unet(
                    mid_latents,
                    t,
                    encoder_hidden_states=condition
                ).sample
            
            # Scheduler step (only update the noisy part, not the conditioning)
            scheduler_output = self.mid_scheduler.step(
                noise_pred, t, mid_latents[:, 4:]
            )
            
            # Update only the noisy part
            mid_latents = torch.cat([mid_latents[:, :4], scheduler_output.prev_sample], dim=1)
        
        # ----------- High-resolution generation -----------
        # Upsample mid-res latents to high-res
        mid_res_latents = mid_latents[:, 4:]  # Get the denoised part
        upsampled_mid_res = self.mid_to_high_upsampler(mid_res_latents)
        
        # Project latent to high-res spatial features (partial, then upsample to save memory)
        high_spatial_latent = self.high_latent_projector(latent)
        high_spatial_latent = high_spatial_latent.reshape(batch_size, 4, 64, 64)
        high_spatial_latent = F.interpolate(high_spatial_latent, size=(256, 256), mode='bilinear')
        
        # Add noise
        high_noise = torch.randn_like(high_spatial_latent)
        high_noisy_latent = self.high_scheduler.add_noise(
            high_spatial_latent,
            high_noise,
            torch.tensor([self.high_scheduler.num_train_timesteps - 1], device=device)
        )
        
        # Apply attention gate to upsampled mid-res features
        attended_mid_res = self.high_attention_gate(upsampled_mid_res, mid_res_latents)
        
        # Concatenate attended mid-res with high-res latents
        high_latents = torch.cat([attended_mid_res, high_noisy_latent], dim=1)
        
        # Set up scheduler
        self.high_scheduler.set_timesteps(num_inference_steps)
        
        # Iterative denoising for high-res
        for t in self.high_scheduler.timesteps:
            # Get model prediction
            with torch.no_grad():
                noise_pred = self.high_unet(
                    high_latents,
                    t,
                    encoder_hidden_states=condition
                ).sample
            
            # Scheduler step (only update the noisy part, not the conditioning)
            scheduler_output = self.high_scheduler.step(
                noise_pred, t, high_latents[:, 4:]
            )
            
            # Update only the noisy part
            high_latents = torch.cat([high_latents[:, :4], scheduler_output.prev_sample], dim=1)
        
        # Decode final latents to image
        final_latents = high_latents[:, 4:]  # Get the denoised part
        images = self.vae.decode(final_latents).sample
        
        return {
            "images": images,
            "mid_latents": mid_res_latents,
            "high_latents": final_latents
        }

class HierarchicalPixelGAN(nn.Module):
    """
    Full Hierarchical PixelGAN model that combines the scaffold and refinement networks
    """
    def __init__(self, latent_dim=1024, condition_dim=1024):
        super().__init__()
        
        # Scaffold network (16x16)
        self.scaffold_net = ScaffoldNetwork(latent_dim, condition_dim)
        
        # Refinement network (64x64 to 256x256)
        self.refinement_net = RefinementNetwork(
            low_res_size=16,
            mid_res_size=64,
            high_res_size=256,
            latent_dim=latent_dim,
            condition_dim=condition_dim
        )
        
    def forward(self, latent, condition):
        """
        Generate image from latent vector and conditioning
        
        Args:
            latent: Input latent vector (B, latent_dim)
            condition: Conditioning information (B, condition_dim)
        """
        # Generate scaffold (low-res structure)
        scaffold_output = self.scaffold_net.generate(latent, condition)
        low_res_latents = scaffold_output["latents"]
        
        # Refine to high-resolution
        refinement_output = self.refinement_net.generate(
            low_res_latents, latent, condition
        )
        
        return {
            "low_res_images": scaffold_output["images"],
            "high_res_images": refinement_output["images"],
            "low_res_latents": low_res_latents,
            "mid_res_latents": refinement_output["mid_latents"],
            "high_res_latents": refinement_output["high_latents"]
        } 