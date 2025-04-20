import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class BiometricEncoder(nn.Module):
    """
    Encoder for biometric feedback signals such as eye tracking
    and sentiment analysis from chat
    """
    def __init__(self, 
                 eye_tracking_dim: int = 16,
                 sentiment_dim: int = 8, 
                 hidden_dim: int = 64,
                 output_dim: int = 32):
        """
        Initialize biometric encoder
        
        Args:
            eye_tracking_dim: Dimension of eye tracking features
            sentiment_dim: Dimension of sentiment analysis features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
        """
        super().__init__()
        
        self.eye_tracking_dim = eye_tracking_dim
        self.sentiment_dim = sentiment_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Eye tracking encoder
        self.eye_encoder = nn.Sequential(
            nn.Linear(eye_tracking_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Sentiment encoder
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, 
                eye_tracking: Optional[torch.Tensor] = None,
                sentiment: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode biometric data into a unified representation
        
        Args:
            eye_tracking: Eye tracking data tensor
            sentiment: Sentiment analysis data tensor
            
        Returns:
            Encoded biometric features
        """
        # Handle missing modalities
        batch_size = eye_tracking.size(0) if eye_tracking is not None else sentiment.size(0)
        device = eye_tracking.device if eye_tracking is not None else sentiment.device
        
        if eye_tracking is None:
            eye_embedding = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            eye_embedding = self.eye_encoder(eye_tracking)
            
        if sentiment is None:
            sentiment_embedding = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            sentiment_embedding = self.sentiment_encoder(sentiment)
            
        # Concatenate and fuse embeddings
        combined = torch.cat([eye_embedding, sentiment_embedding], dim=1)
        return self.fusion(combined)


class CriticNetwork(nn.Module):
    """
    Critic network for predicting user satisfaction based on biometric feedback
    and image features
    """
    def __init__(self, 
                 image_feature_dim: int = 1024,
                 biometric_dim: int = 32,
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        """
        Initialize critic network
        
        Args:
            image_feature_dim: Dimension of image features
            biometric_dim: Dimension of biometric embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.image_feature_dim = image_feature_dim
        self.biometric_dim = biometric_dim
        
        # Image feature projection
        self.image_projector = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Biometric feature projection
        self.biometric_projector = nn.Sequential(
            nn.Linear(biometric_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined analysis
        self.satisfaction_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single satisfaction score
        )
        
        # Classification head for high-level feedback categories
        self.feedback_classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)  # 5 feedback categories (style, content, detail, color, composition)
        )
        
    def forward(self, 
                image_features: torch.Tensor,
                biometric_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict user satisfaction and feedback categories
        
        Args:
            image_features: Features from the generated image
            biometric_embedding: Encoded biometric feedback
            
        Returns:
            Dictionary with satisfaction score and feedback category probabilities
        """
        # Project features
        img_proj = self.image_projector(image_features)
        bio_proj = self.biometric_projector(biometric_embedding)
        
        # Combine features
        combined = torch.cat([img_proj, bio_proj], dim=1)
        
        # Predict satisfaction score (0 to 1)
        satisfaction = torch.sigmoid(self.satisfaction_predictor(combined))
        
        # Predict feedback categories (softmax over categories)
        feedback_logits = self.feedback_classifier(combined)
        feedback_probs = F.softmax(feedback_logits, dim=1)
        
        return {
            "satisfaction": satisfaction,
            "feedback_logits": feedback_logits,
            "feedback_probs": feedback_probs
        }


class RefinerNetwork(nn.Module):
    """
    Refiner network for optimizing outputs using Proximal Policy Optimization (PPO)
    against the Critic's predictions
    """
    def __init__(self,
                 latent_dim: int = 1024,
                 condition_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize refiner network
        
        Args:
            latent_dim: Dimension of latent vectors to refine
            condition_dim: Dimension of conditioning information
            hidden_dim: Dimension of hidden layers
            num_layers: Number of refinement layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Initial projection layers
        self.latent_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Refinement layers (with residual connections)
        self.refinement_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )
            self.refinement_layers.append(layer)
            
        # Final projection back to latent space
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Policy head for PPO
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
        )
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent: torch.Tensor, condition: torch.Tensor, 
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Refine latent vectors for improved image generation
        
        Args:
            latent: Input latent vectors to refine
            condition: Conditioning information
            deterministic: Whether to use deterministic sampling
            
        Returns:
            Dictionary with refined latents and policy information
        """
        # Project inputs
        latent_features = self.latent_projector(latent)
        condition_features = self.condition_projector(condition)
        
        # Combine features
        combined = torch.cat([latent_features, condition_features], dim=1)
        
        # Apply refinement layers with residual connections
        x = combined
        for layer in self.refinement_layers:
            residual = x
            x = layer(x) + residual
            
        # Get policy parameters
        policy_params = self.policy_head(x)
        mean, log_std = torch.chunk(policy_params, 2, dim=1)
        log_std = torch.clamp(log_std, -20, 2)  # Constrain for stability
        std = torch.exp(log_std)
        
        # Sample from policy distribution or use mean
        if deterministic:
            refined_latent = mean
        else:
            # Reparameterization trick
            normal = torch.randn_like(mean)
            refined_latent = mean + normal * std
            
        # Final projection
        final_latent = self.output_projector(x)
        
        # Value prediction for PPO
        value = self.value_head(x)
        
        return {
            "refined_latent": refined_latent,
            "final_latent": final_latent,
            "mean": mean,
            "log_std": log_std,
            "std": std,
            "value": value
        }


class CriticRefinerSystem:
    """
    Complete Critic-Refiner system with PPO-based training
    """
    def __init__(self,
                 critic: CriticNetwork,
                 refiner: RefinerNetwork,
                 biometric_encoder: BiometricEncoder,
                 learning_rate: float = 0.0001,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize the Critic-Refiner system
        
        Args:
            critic: CriticNetwork instance
            refiner: RefinerNetwork instance
            biometric_encoder: BiometricEncoder instance
            learning_rate: Learning rate for optimizers
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient for exploration
        """
        self.critic = critic
        self.refiner = refiner
        self.biometric_encoder = biometric_encoder
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizers
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
        self.refiner_optimizer = optim.Adam(refiner.parameters(), lr=learning_rate)
        self.biometric_optimizer = optim.Adam(biometric_encoder.parameters(), lr=learning_rate)
        
        # Training state
        self.train_iterations = 0
        
    def compute_advantages(self, 
                          rewards: torch.Tensor, 
                          values: torch.Tensor, 
                          dones: torch.Tensor,
                          gamma: float = 0.99, 
                          lam: float = 0.95) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Batch of rewards
            values: Batch of value predictions
            dones: Batch of done flags
            gamma: Discount factor
            lam: GAE parameter
            
        Returns:
            Computed advantages
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Reverse iteration through time steps
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            # Compute TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute GAE
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def train_critic(self, 
                    image_features: torch.Tensor,
                    biometric_features: Dict[str, torch.Tensor],
                    satisfaction_scores: torch.Tensor,
                    feedback_categories: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train the critic network to predict user satisfaction
        
        Args:
            image_features: Image features from generator
            biometric_features: Biometric feedback features
            satisfaction_scores: Ground truth satisfaction scores
            feedback_categories: Optional ground truth feedback categories
            
        Returns:
            Dictionary with loss metrics
        """
        # Zero gradients
        self.critic_optimizer.zero_grad()
        self.biometric_optimizer.zero_grad()
        
        # Encode biometric features
        biometric_embedding = self.biometric_encoder(
            eye_tracking=biometric_features.get("eye_tracking"),
            sentiment=biometric_features.get("sentiment")
        )
        
        # Get critic predictions
        critic_output = self.critic(image_features, biometric_embedding)
        
        # Compute satisfaction loss (MSE)
        satisfaction_loss = F.mse_loss(
            critic_output["satisfaction"].squeeze(-1),
            satisfaction_scores
        )
        
        # Compute feedback category loss if available (cross-entropy)
        if feedback_categories is not None:
            feedback_loss = F.cross_entropy(
                critic_output["feedback_logits"],
                feedback_categories
            )
        else:
            feedback_loss = torch.tensor(0.0, device=satisfaction_scores.device)
            
        # Total loss
        total_loss = satisfaction_loss + feedback_loss
        
        # Backward and optimize
        total_loss.backward()
        self.critic_optimizer.step()
        self.biometric_optimizer.step()
        
        return {
            "satisfaction_loss": satisfaction_loss.item(),
            "feedback_loss": feedback_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def update_refiner_ppo(self,
                          old_latents: torch.Tensor,
                          old_conditions: torch.Tensor,
                          old_means: torch.Tensor,
                          old_log_stds: torch.Tensor,
                          old_values: torch.Tensor,
                          advantages: torch.Tensor,
                          returns: torch.Tensor,
                          epochs: int = 4,
                          batch_size: int = 64) -> Dict[str, float]:
        """
        Update the refiner network using PPO
        
        Args:
            old_latents: Batch of original latent vectors
            old_conditions: Batch of conditioning information
            old_means: Batch of old policy means
            old_log_stds: Batch of old policy log stds
            old_values: Batch of old value predictions
            advantages: Computed advantages
            returns: Computed returns
            epochs: Number of PPO epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary with training metrics
        """
        dataset_size = old_latents.size(0)
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for _ in range(epochs):
            # Generate random indices
            indices = torch.randperm(dataset_size)
            
            # Mini-batch iterations
            for start_idx in range(0, dataset_size, batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                batch_latents = old_latents[idx]
                batch_conditions = old_conditions[idx]
                batch_means = old_means[idx]
                batch_log_stds = old_log_stds[idx]
                batch_values = old_values[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Forward pass
                outputs = self.refiner(batch_latents, batch_conditions)
                
                # Get new policy distribution
                new_means = outputs["mean"]
                new_log_stds = outputs["log_std"]
                new_values = outputs["value"].squeeze(-1)
                
                # Compute log probabilities for old and new policies
                old_distribution = torch.distributions.Normal(batch_means, torch.exp(batch_log_stds))
                new_distribution = torch.distributions.Normal(new_means, torch.exp(new_log_stds))
                
                # Get action (the refined latent is the action)
                action = outputs["refined_latent"]
                
                # Compute log probabilities
                old_log_probs = old_distribution.log_prob(action).sum(dim=1)
                new_log_probs = new_distribution.log_prob(action).sum(dim=1)
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Compute PPO losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = self.value_coef * F.mse_loss(new_values, batch_returns)
                
                # Compute entropy for exploration
                entropy_loss = -self.entropy_coef * new_distribution.entropy().mean()
                
                # Total loss
                loss = policy_loss + value_loss + entropy_loss
                
                # Zero gradients and backward
                self.refiner_optimizer.zero_grad()
                loss.backward()
                self.refiner_optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                
        # Increment training iterations
        self.train_iterations += 1
                
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses)
        }
    
    def refine_image_latent(self, 
                           latent: torch.Tensor, 
                           condition: torch.Tensor, 
                           deterministic: bool = False) -> torch.Tensor:
        """
        Refine a latent vector for improved image generation
        
        Args:
            latent: Original latent vector
            condition: Conditioning information
            deterministic: Whether to use deterministic sampling
            
        Returns:
            Refined latent vector
        """
        with torch.no_grad():
            outputs = self.refiner(latent, condition, deterministic=deterministic)
            return outputs["refined_latent"]
    
    def predict_satisfaction(self, 
                            image_features: torch.Tensor,
                            biometric_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict user satisfaction from image and biometric features
        
        Args:
            image_features: Features from the generated image
            biometric_features: Biometric feedback features
            
        Returns:
            Dictionary with satisfaction score and feedback probabilities
        """
        with torch.no_grad():
            # Encode biometric features
            biometric_embedding = self.biometric_encoder(
                eye_tracking=biometric_features.get("eye_tracking"),
                sentiment=biometric_features.get("sentiment")
            )
            
            # Get critic predictions
            return self.critic(image_features, biometric_embedding)
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            "critic": self.critic.state_dict(),
            "refiner": self.refiner.state_dict(),
            "biometric_encoder": self.biometric_encoder.state_dict(),
            "iterations": self.train_iterations
        }, path)
        
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.critic.load_state_dict(checkpoint["critic"])
        self.refiner.load_state_dict(checkpoint["refiner"])
        self.biometric_encoder.load_state_dict(checkpoint["biometric_encoder"])
        self.train_iterations = checkpoint["iterations"] 