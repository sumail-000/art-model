import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

class DynamicIntentParser(nn.Module):
    """
    Dynamic intent parser using Few-Shot Learning and RLHF
    for refining command interpretation
    """
    
    def __init__(self, 
                 base_model_name="google/flan-t5-xl",
                 max_input_length=512,
                 max_output_length=128,
                 embedding_dim=1024):
        super().__init__()
        
        # Base T5 model for intent parsing
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        self.base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        # Few-shot in-context learning parameters
        self.max_exemplars = 5  # Max number of few-shot examples to include
        self.exemplar_memory = []  # List to store few-shot examples
        
        # RL components
        self.value_head = nn.Sequential(
            nn.Linear(self.base_model.config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Scalar value prediction
        )
        
        # Latent projection for multimodal embeddings
        self.multimodal_projection = nn.Linear(embedding_dim, self.base_model.config.d_model)
        
        # Parameters
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def _format_few_shot_prompt(self, input_text):
        """Format input with few-shot examples"""
        prompt = ""
        
        # Add exemplars as few-shot context
        for example in self.exemplar_memory[-self.max_exemplars:]:
            prompt += f"Input: {example['input']}\nIntent: {example['intent']}\n\n"
            
        # Add current input
        prompt += f"Input: {input_text}\nIntent:"
        
        return prompt
    
    def parse_intent(self, input_text, multimodal_embedding=None, temperature=0.7):
        """Parse the user's intent from text input and optional multimodal context"""
        
        # Format with few-shot examples
        prompt = self._format_few_shot_prompt(input_text)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               max_length=self.max_input_length,
                               truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Incorporate multimodal embedding if available
        if multimodal_embedding is not None:
            # Project to model dimension
            modal_features = self.multimodal_projection(multimodal_embedding)
            
            # Add as prefix to the encoder hidden states
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Prepend multimodal features to encoder outputs
            encoder_hidden_states = encoder_outputs.last_hidden_state
            extended_hidden_states = torch.cat(
                [modal_features.unsqueeze(1), encoder_hidden_states], dim=1
            )
            
            # Extend attention mask for the new token
            extended_attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1, device=attention_mask.device),
                 attention_mask], 
                dim=1
            )
            
            # Generate with custom encoder states
            outputs = self.base_model.generate(
                encoder_outputs=(extended_hidden_states,),
                attention_mask=extended_attention_mask,
                max_length=self.max_output_length,
                temperature=temperature,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True
            )
        else:
            # Standard generation without multimodal context
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_output_length,
                temperature=temperature,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode the generated intent
        intent = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Get value prediction for RL
        if multimodal_embedding is not None:
            # Use the extended hidden states
            value = self.value_head(extended_hidden_states.mean(dim=1))
        else:
            # Get encoder outputs first
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            value = self.value_head(encoder_outputs.last_hidden_state.mean(dim=1))
        
        return {
            "input": input_text,
            "intent": intent,
            "value": value.item(),
            "logits": outputs.scores
        }
    
    def add_example(self, input_text, intent, reward=None):
        """Add a new few-shot example to memory"""
        example = {
            "input": input_text,
            "intent": intent,
            "reward": reward
        }
        
        # Sort by reward if available, keep highest reward examples
        if reward is not None and len(self.exemplar_memory) >= self.max_exemplars:
            self.exemplar_memory.append(example)
            self.exemplar_memory = sorted(
                self.exemplar_memory, 
                key=lambda x: x.get("reward", 0),
                reverse=True
            )[:self.max_exemplars]
        else:
            # Simple FIFO if no reward available
            if len(self.exemplar_memory) >= self.max_exemplars:
                self.exemplar_memory.pop(0)
            self.exemplar_memory.append(example)
            
    def update_from_feedback(self, old_outputs, rewards, optimizer, gamma=0.99):
        """Update model using RLHF with PPO"""
        # This would contain PPO-specific implementation
        # Simplified version shown here
        
        # Compute advantages
        values = torch.tensor([output["value"] for output in old_outputs])
        rewards = torch.tensor(rewards)
        advantages = rewards - values
        
        # Compute policy loss
        policy_loss = 0
        value_loss = 0
        
        # Process each example
        for i, (output, advantage) in enumerate(zip(old_outputs, advantages)):
            # Get log probs for the chosen actions
            logits = output["logits"]
            
            # Compute policy gradient loss
            policy_loss += -advantage * F.log_softmax(logits, dim=-1)
            
            # Compute value loss
            value_loss += F.mse_loss(values[i], rewards[i])
            
            # Add example to memory with its reward
            self.add_example(output["input"], output["intent"], rewards[i].item())
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item() 