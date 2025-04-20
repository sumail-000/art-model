import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer

class PixelArtDataset(Dataset):
    """
    Dataset for pixel art images with text descriptions
    """
    
    def __init__(self, 
                 sources, 
                 split="train",
                 resolution=256,
                 max_text_length=77,
                 augmentation=True):
        """
        Initialize the dataset
        
        Args:
            sources: List of data source configurations
            split: Data split ('train', 'val', or 'test')
            resolution: Image resolution
            max_text_length: Maximum text length for tokenization
            augmentation: Whether to apply data augmentation
        """
        self.sources = sources
        self.split = split
        self.resolution = resolution
        self.max_text_length = max_text_length
        self.augmentation = augmentation
        
        # Initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Setup image transforms
        if augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        # Load dataset
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset from sources"""
        samples = []
        
        for source in self.sources:
            source_path = source["path"]
            metadata_path = os.path.join(source_path, f"{self.split}_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    source_samples = json.load(f)
                    
                # Apply source weight
                source_weight = source.get("weight", 1.0)
                if source_weight < 1.0:
                    # Randomly sample based on weight
                    num_samples = int(len(source_samples) * source_weight)
                    indices = np.random.choice(len(source_samples), num_samples, replace=False)
                    source_samples = [source_samples[i] for i in indices]
                
                # Add source to sample paths
                for sample in source_samples:
                    if "image_path" in sample:
                        sample["image_path"] = os.path.join(source_path, sample["image_path"])
                    if "mask_path" in sample:
                        sample["mask_path"] = os.path.join(source_path, sample["mask_path"])
                
                samples.extend(source_samples)
        
        return samples
    
    def _tokenize_text(self, text):
        """Tokenize text description"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "text_input_ids": tokens.input_ids[0],
            "text_attention_mask": tokens.attention_mask[0]
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        sample = self.samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        
        # Load segmentation mask if available
        mask = None
        if "mask_path" in sample and os.path.exists(sample["mask_path"]):
            mask_path = sample["mask_path"]
            mask = Image.open(mask_path).convert("L")
            mask = transforms.Resize((self.resolution, self.resolution))(mask)
            mask = transforms.ToTensor()(mask)
        
        # Get text description
        text = sample.get("description", "pixel art")
        text_tokens = self._tokenize_text(text)
        
        # Get target intent if available
        target_intent = sample.get("target_intent", None)
        target_intent_tokens = self._tokenize_text(target_intent) if target_intent else {}
        
        # Get style embedding if available
        style_embedding = None
        if "style_embedding" in sample:
            style_embedding = torch.tensor(sample["style_embedding"])
        
        # Get user feedback if available
        feedback = sample.get("feedback", None)
        
        # Combine everything
        item = {
            "pixel_values": pixel_values,
            **text_tokens
        }
        
        # Add optional fields if available
        if mask is not None:
            item["mask"] = mask
            
        if target_intent_tokens:
            item["target_intent_ids"] = target_intent_tokens["text_input_ids"]
            
        if style_embedding is not None:
            item["style_embedding"] = style_embedding
            
        if feedback:
            item["feedback"] = feedback
            
        return item 