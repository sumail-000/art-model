import torch
import torch.nn as nn
import transformers
from torch.nn import functional as F

class CLIPLikeEncoder(nn.Module):
    """
    CLIP-like encoder for mapping text, image, and voice inputs to a unified latent space
    """
    
    def __init__(self, 
                 embed_dim=1024, 
                 text_model_name="openai/clip-vit-large-patch14",
                 vision_model_name="openai/clip-vit-large-patch14",
                 audio_model_name="openai/whisper-large-v3"):
        super().__init__()
        
        # Text encoder from HuggingFace transformers
        self.text_encoder = transformers.CLIPTextModel.from_pretrained(text_model_name)
        
        # Vision encoder from HuggingFace transformers
        self.vision_encoder = transformers.CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Audio encoder (using Whisper as foundation)
        self.audio_encoder = transformers.WhisperModel.from_pretrained(audio_model_name)
        
        # Sketch encoder (re-uses vision encoder with special preprocessing)
        self.sketch_preprocessor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )
        
        # Projection layers to unified latent space
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, embed_dim)
        self.audio_projection = nn.Linear(self.audio_encoder.config.d_model, embed_dim)
        
        # Learnable modality tokens (to help distinguish which modality the embedding came from)
        self.modality_tokens = nn.Parameter(torch.randn(3, embed_dim))
        
    def encode_text(self, text_input_ids, attention_mask=None):
        text_features = self.text_encoder(input_ids=text_input_ids, 
                                         attention_mask=attention_mask).pooler_output
        text_embedding = self.text_projection(text_features)
        # Add modality token (0 = text)
        text_embedding = text_embedding + self.modality_tokens[0]
        return F.normalize(text_embedding, dim=-1)
        
    def encode_image(self, pixel_values):
        vision_features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        vision_embedding = self.vision_projection(vision_features)
        # Add modality token (1 = image)
        vision_embedding = vision_embedding + self.modality_tokens[1]
        return F.normalize(vision_embedding, dim=-1)
    
    def encode_audio(self, audio_features):
        audio_output = self.audio_encoder(audio_features).last_hidden_state.mean(dim=1)
        audio_embedding = self.audio_projection(audio_output)
        # Add modality token (2 = audio)
        audio_embedding = audio_embedding + self.modality_tokens[2]
        return F.normalize(audio_embedding, dim=-1)
    
    def encode_sketch(self, sketch):
        # Preprocess sketch (assumed to be single-channel)
        processed_sketch = self.sketch_preprocessor(sketch)
        # Use the same vision encoder for sketch processing
        return self.encode_image(processed_sketch)
    
    def forward(self, batch):
        """Process a batch with any combination of modalities"""
        embeddings = []
        
        if "text_input_ids" in batch:
            text_embedding = self.encode_text(
                batch["text_input_ids"], 
                attention_mask=batch.get("text_attention_mask")
            )
            embeddings.append(text_embedding)
            
        if "pixel_values" in batch:
            vision_embedding = self.encode_image(batch["pixel_values"])
            embeddings.append(vision_embedding)
            
        if "audio_features" in batch:
            audio_embedding = self.encode_audio(batch["audio_features"])
            embeddings.append(audio_embedding)
            
        if "sketch" in batch:
            sketch_embedding = self.encode_sketch(batch["sketch"])
            embeddings.append(sketch_embedding)
            
        # Return all embeddings stacked
        return torch.stack(embeddings, dim=1) 