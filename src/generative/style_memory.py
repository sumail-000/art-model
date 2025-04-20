import torch
import torch.nn as nn
import faiss
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import redis
import json

@dataclass
class StyleItem:
    """An item in the style memory bank"""
    embedding: np.ndarray  # Vector representation of the style
    metadata: Dict  # Additional information (user, timestamp, description)
    example_image_path: Optional[str] = None  # Path to an example image with this style

class StyleMemoryBank:
    """
    Vector database for storing and retrieving style embeddings and preferences
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024, 
                 redis_url: Optional[str] = None,
                 index_path: Optional[str] = None,
                 use_redis_cache: bool = True):
        """
        Initialize style memory bank
        
        Args:
            embedding_dim: Dimensionality of style embeddings
            redis_url: URL for Redis connection (optional)
            index_path: Path to load existing FAISS index
            use_redis_cache: Whether to use Redis for fast lookup caching
        """
        self.embedding_dim = embedding_dim
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity
        
        # Add HNSW index for faster search (hierarchical navigable small world graph)
        # Good balance between search speed and accuracy
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbors per node
        self.index.hnsw.efConstruction = 128  # Higher for more accurate index construction
        self.index.hnsw.efSearch = 64  # Higher for more accurate search
        
        # Storage for metadata (FAISS only stores vectors)
        self.metadata_store = {}  # id -> StyleItem
        self.next_id = 0
        
        # Optional Redis client for fast lookup caching
        self.redis_client = None
        self.use_redis_cache = use_redis_cache
        if redis_url and use_redis_cache:
            self.redis_client = redis.Redis.from_url(redis_url)
            
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_style(self, embedding: torch.Tensor, metadata: Dict, example_image_path: Optional[str] = None) -> int:
        """
        Add a style embedding to the memory bank
        
        Args:
            embedding: Style embedding tensor
            metadata: Dict with style metadata (user_id, description, timestamp, etc.)
            example_image_path: Optional path to an example image
            
        Returns:
            style_id: ID of the added style
        """
        # Convert to numpy if tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
            
        # Ensure we have a single vector of the right shape
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)  # Remove batch dimension if present
            
        # Create style item
        style_item = StyleItem(
            embedding=embedding,
            metadata=metadata,
            example_image_path=example_image_path
        )
        
        # Add to FAISS index
        style_id = self.next_id
        self.index.add(np.array([embedding]))  # FAISS requires 2D array
        
        # Store metadata
        self.metadata_store[style_id] = style_item
        
        # Cache in Redis if enabled
        if self.redis_client:
            cache_key = f"style:{style_id}"
            # Store user_id for quick access patterns
            if "user_id" in metadata:
                user_key = f"user:{metadata['user_id']}:styles"
                self.redis_client.sadd(user_key, style_id)
            
            # Store serialized metadata
            self.redis_client.set(
                cache_key, 
                json.dumps({
                    "metadata": metadata,
                    "example_image_path": example_image_path
                })
            )
        
        self.next_id += 1
        return style_id
        
    def search(self, 
               query_embedding: torch.Tensor, 
               k: int = 5, 
               filter_user_id: Optional[str] = None) -> List[Tuple[int, float, StyleItem]]:
        """
        Search for similar styles
        
        Args:
            query_embedding: Query embedding tensor
            k: Number of results to return
            filter_user_id: Optional user ID to filter results
            
        Returns:
            List of tuples (style_id, distance, style_item)
        """
        # Convert to numpy if tensor
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
            
        # Ensure we have a single vector of the right shape
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze(0)
            
        # Filter by user if requested
        if filter_user_id and self.redis_client:
            # Get style ids for this user from Redis
            user_key = f"user:{filter_user_id}:styles"
            user_style_ids = self.redis_client.smembers(user_key)
            
            if user_style_ids:
                # Convert to integers
                user_style_ids = [int(sid) for sid in user_style_ids]
                
                # Get embeddings for these styles
                embeddings = []
                for sid in user_style_ids:
                    if sid in self.metadata_store:
                        embeddings.append(self.metadata_store[sid].embedding)
                
                if embeddings:
                    # Stack into a single array
                    embeddings = np.vstack(embeddings)
                    
                    # Compute distances
                    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
                    
                    # Sort by distance
                    indices = np.argsort(distances)[:k]
                    
                    # Return results
                    return [(user_style_ids[i], distances[i], self.metadata_store[user_style_ids[i]]) 
                            for i in indices]
        
        # Standard search using FAISS
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS may return -1 if not enough results
                continue
                
            if idx in self.metadata_store:
                results.append((idx, distance, self.metadata_store[idx]))
                
        return results
    
    def get_style(self, style_id: int) -> Optional[StyleItem]:
        """Get a specific style by ID"""
        # Try Redis cache first if enabled
        if self.redis_client:
            cache_key = f"style:{style_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                cached_data = json.loads(cached_data)
                
                # Only metadata is stored in Redis, need to get embedding from FAISS
                if style_id in self.metadata_store:
                    return self.metadata_store[style_id]
        
        # Fall back to direct lookup
        return self.metadata_store.get(style_id)
    
    def get_user_styles(self, user_id: str) -> List[Tuple[int, StyleItem]]:
        """Get all styles for a specific user"""
        user_styles = []
        
        # Try Redis first if enabled
        if self.redis_client:
            user_key = f"user:{user_id}:styles"
            style_ids = self.redis_client.smembers(user_key)
            
            if style_ids:
                for sid in style_ids:
                    sid = int(sid)
                    if sid in self.metadata_store:
                        user_styles.append((sid, self.metadata_store[sid]))
        
        # Fall back to scanning all styles
        if not user_styles:
            for style_id, style_item in self.metadata_store.items():
                if style_item.metadata.get("user_id") == user_id:
                    user_styles.append((style_id, style_item))
                    
        return user_styles
    
    def save(self, path: str):
        """Save the memory bank to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "next_id": self.next_id,
                "embedding_dim": self.embedding_dim
            }, f)
            
    def load(self, path: str):
        """Load the memory bank from disk"""
        # Load FAISS index if it exists
        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata if it exists
        if os.path.exists(f"{path}.meta"):
            with open(f"{path}.meta", "rb") as f:
                data = pickle.load(f)
                self.metadata_store = data["metadata_store"]
                self.next_id = data["next_id"]
                self.embedding_dim = data["embedding_dim"]

class StyleEmbedder(nn.Module):
    """
    Neural network for embedding images into style vectors
    """
    def __init__(self, 
                 input_dim: int = 2048, 
                 hidden_dim: int = 1024, 
                 output_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, image_features):
        """
        Extract style embedding from image features
        
        Args:
            image_features: Image features from encoder
            
        Returns:
            style_embedding: Normalized style embedding vector
        """
        style_embedding = self.network(image_features)
        # Normalize to unit length for consistent similarity calculations
        return torch.nn.functional.normalize(style_embedding, p=2, dim=1) 