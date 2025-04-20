import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class HeadController(nn.Module):
    """
    Controller for read/write heads in the Neural Turing Machine
    """
    def __init__(self, 
                 hidden_size: int,
                 memory_width: int,
                 num_heads: int = 1,
                 addressing_mode: str = "content_and_location"):
        """
        Initialize head controller
        
        Args:
            hidden_size: Size of hidden state in controller
            memory_width: Width of memory vectors
            num_heads: Number of read/write heads
            addressing_mode: How to address memory ('content', 'location', or 'content_and_location')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_width = memory_width
        self.num_heads = num_heads
        self.addressing_mode = addressing_mode
        
        # Parameter sizes
        self.key_size = memory_width
        self.shift_range = 3  # Previous, current, and next location
        
        # Parameters for each head
        # - key: vector to match memory contents
        # - beta: key strength (for content addressing)
        # - gate: interpolation between previous and new weights
        # - shift: convolutional shift [-1, 0, 1]
        # - gamma: sharpening factor
        # - erase: erase vector (for write heads)
        # - add: add vector (for write heads)
        
        # Linear layer sizes
        if addressing_mode == "content":
            parameters_per_head = memory_width + 1  # key + beta
        elif addressing_mode == "location":
            parameters_per_head = 1 + self.shift_range + 1  # gate + shift + gamma
        else:  # content_and_location
            parameters_per_head = memory_width + 1 + 1 + self.shift_range + 1
            
        # For write heads, add erase and add vectors
        self.write_head_extra = 2 * memory_width
        
        # Parameters for read and write heads
        self.read_param_size = parameters_per_head
        self.write_param_size = parameters_per_head + self.write_head_extra
        
        # Linear layers to generate head parameters
        self.read_heads_linear = nn.Linear(
            hidden_size, self.read_param_size * num_heads
        )
        self.write_heads_linear = nn.Linear(
            hidden_size, self.write_param_size * num_heads
        )
        
    def address_content(self, key, beta, memory):
        """Content-based addressing using cosine similarity"""
        # Shape: (batch_size, 1, memory_width) x (batch_size, memory_slots, memory_width)
        similarity = F.cosine_similarity(
            key.unsqueeze(1), memory, dim=2
        )
        
        # Apply key strength (beta)
        similarity = similarity * beta.unsqueeze(1)
        
        # Softmax to get weights summing to 1
        return F.softmax(similarity, dim=1)
    
    def address_location(self, prev_weights, gate, shift, gamma):
        """Location-based addressing with shifting and sharpening"""
        # Interpolate between previous and new weights
        gated_weights = gate.unsqueeze(1) * prev_weights
        
        # Convolutional shift
        batch_size = prev_weights.size(0)
        memory_slots = prev_weights.size(1)
        
        # Create circular convolution kernel
        shift_kernel = torch.zeros(
            batch_size, memory_slots, self.shift_range, 
            device=prev_weights.device
        )
        
        # Fill shift kernel based on shift vector
        for b in range(batch_size):
            for i in range(memory_slots):
                for j in range(self.shift_range):
                    shift_kernel[b, i, j] = shift[b, j]
        
        # Circular convolution (implemented as FFT-based convolution)
        shifted_weights = torch.zeros_like(prev_weights)
        for b in range(batch_size):
            # For each memory position, apply the shift
            for pos in range(memory_slots):
                # Calculate shifted positions (with circular wrapping)
                left_pos = (pos - 1) % memory_slots
                right_pos = (pos + 1) % memory_slots
                
                # Apply shift weights
                shifted_weights[b, pos] = (
                    shift_kernel[b, pos, 0] * prev_weights[b, left_pos] +
                    shift_kernel[b, pos, 1] * prev_weights[b, pos] +
                    shift_kernel[b, pos, 2] * prev_weights[b, right_pos]
                )
        
        # Sharpening
        sharp_weights = shifted_weights ** gamma.unsqueeze(1)
        sharp_weights = sharp_weights / (sharp_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        return sharp_weights
    
    def address_memory(self, 
                      controller_state, 
                      prev_weights, 
                      memory, 
                      head_type='read'):
        """
        Address memory using content and/or location addressing
        
        Args:
            controller_state: Hidden state from controller
            prev_weights: Previous addressing weights
            memory: Current memory state
            head_type: 'read' or 'write'
            
        Returns:
            weights: Addressing weights for each head
            parameters: Extracted parameters (e.g., erase & add for write)
        """
        batch_size = memory.size(0)
        memory_slots = memory.size(1)
        
        # Get parameters from controller state
        if head_type == 'read':
            params = self.read_heads_linear(controller_state)
            params = params.view(batch_size, self.num_heads, -1)
        else:  # write
            params = self.write_heads_linear(controller_state)
            params = params.view(batch_size, self.num_heads, -1)
            
        # Extract head parameters
        head_weights = []
        head_params = []
        
        for h in range(self.num_heads):
            head_param = params[:, h]
            
            if self.addressing_mode in ['content', 'content_and_location']:
                # Extract content addressing parameters
                key = head_param[:, :self.key_size]
                beta = F.softplus(head_param[:, self.key_size])
                
                # Content-based addressing
                content_weights = self.address_content(key, beta, memory)
                
                if self.addressing_mode == 'content':
                    weights = content_weights
                    param_offset = self.key_size + 1
                else:
                    # Extract location addressing parameters
                    gate = torch.sigmoid(head_param[:, self.key_size + 1])
                    shift = F.softmax(
                        head_param[:, self.key_size + 2:self.key_size + 2 + self.shift_range],
                        dim=1
                    )
                    gamma = 1 + F.softplus(head_param[:, self.key_size + 2 + self.shift_range])
                    
                    # Apply gating between content and previous weights
                    gated_weights = gate.unsqueeze(1) * content_weights + (1 - gate.unsqueeze(1)) * prev_weights[:, h]
                    
                    # Apply location addressing
                    weights = self.address_location(gated_weights, gate, shift, gamma)
                    param_offset = self.key_size + 2 + self.shift_range + 1
            else:  # location only
                # Extract location addressing parameters
                gate = torch.sigmoid(head_param[:, 0])
                shift = F.softmax(head_param[:, 1:1 + self.shift_range], dim=1)
                gamma = 1 + F.softplus(head_param[:, 1 + self.shift_range])
                
                # Apply location addressing
                weights = self.address_location(prev_weights[:, h], gate, shift, gamma)
                param_offset = 1 + self.shift_range + 1
            
            # For write heads, extract erase and add vectors
            if head_type == 'write':
                erase = torch.sigmoid(head_param[:, param_offset:param_offset + self.memory_width])
                add = torch.tanh(head_param[:, param_offset + self.memory_width:])
                head_params.append((erase, add))
                
            head_weights.append(weights)
            
        # Stack weights for all heads
        stacked_weights = torch.stack(head_weights, dim=1)
        
        return stacked_weights, head_params if head_type == 'write' else None


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine implementation for long-term memory
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 memory_slots: int = 128,
                 memory_width: int = 64,
                 num_read_heads: int = 1,
                 num_write_heads: int = 1,
                 addressing_mode: str = "content_and_location",
                 controller_type: str = "lstm"):
        """
        Initialize Neural Turing Machine
        
        Args:
            input_size: Size of input vector
            hidden_size: Size of controller hidden state
            output_size: Size of output vector
            memory_slots: Number of memory slots
            memory_width: Width of each memory slot
            num_read_heads: Number of read heads
            num_write_heads: Number of write heads
            addressing_mode: Memory addressing mode
            controller_type: Type of controller network
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_slots = memory_slots
        self.memory_width = memory_width
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.addressing_mode = addressing_mode
        
        # Memory shape: (batch_size, memory_slots, memory_width)
        
        # Read data size (input + read content from previous time step)
        self.read_data_size = input_size + num_read_heads * memory_width
        
        # Controller network
        if controller_type == "lstm":
            self.controller = nn.LSTMCell(
                self.read_data_size, hidden_size
            )
        else:  # Simple RNN
            self.controller = nn.RNNCell(
                self.read_data_size, hidden_size
            )
        
        # Memory addressing
        self.read_heads = HeadController(
            hidden_size, memory_width, num_read_heads, addressing_mode
        )
        
        self.write_heads = HeadController(
            hidden_size, memory_width, num_write_heads, addressing_mode
        )
        
        # Output layer
        self.output_layer = nn.Linear(
            hidden_size + num_read_heads * memory_width, output_size
        )
        
        # Initialize memory bias to be orthogonal for better performance
        self.register_buffer(
            'mem_bias', torch.zeros(memory_slots, memory_width)
        )
        # Initialize with small random values
        nn.init.xavier_uniform_(self.mem_bias)
        
    def init_state(self, batch_size, device=None):
        """Initialize NTM state"""
        # Initial memory
        memory = torch.stack(
            [self.mem_bias for _ in range(batch_size)]
        )
        if device:
            memory = memory.to(device)
        
        # Initial read and write weights (uniform across all memory)
        read_weights = torch.zeros(
            batch_size, self.num_read_heads, self.memory_slots,
            device=device
        )
        read_weights[:, :, 0] = 1.0  # Focus on first location initially
        
        write_weights = torch.zeros(
            batch_size, self.num_write_heads, self.memory_slots,
            device=device
        )
        write_weights[:, :, 0] = 1.0  # Focus on first location initially
        
        # Initial read vectors (zeros)
        read_vectors = torch.zeros(
            batch_size, self.num_read_heads * self.memory_width,
            device=device
        )
        
        # Initial controller state
        if isinstance(self.controller, nn.LSTMCell):
            controller_state = (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device)
            )
        else:
            controller_state = torch.zeros(
                batch_size, self.hidden_size, device=device
            )
            
        return {
            'memory': memory,
            'read_weights': read_weights,
            'write_weights': write_weights,
            'read_vectors': read_vectors,
            'controller_state': controller_state
        }
    
    def read_memory(self, memory, read_weights):
        """Read from memory using attention weights"""
        # read_weights: (batch_size, num_heads, memory_slots)
        # memory: (batch_size, memory_slots, memory_width)
        
        # For each head, compute weighted sum of memory
        read_vectors = []
        
        for head in range(self.num_read_heads):
            # Extract weights for this head
            head_weights = read_weights[:, head].unsqueeze(2)
            # Apply weights to memory
            read_vector = torch.bmm(
                memory.transpose(1, 2),  # (batch_size, memory_width, memory_slots)
                head_weights  # (batch_size, memory_slots, 1)
            ).squeeze(2)  # (batch_size, memory_width)
            
            read_vectors.append(read_vector)
            
        # Concatenate all read vectors
        return torch.cat(read_vectors, dim=1)
    
    def write_memory(self, memory, write_weights, write_params):
        """Write to memory using attention weights"""
        # write_weights: (batch_size, num_heads, memory_slots)
        # write_params: list of (erase, add) tuples for each head
        # memory: (batch_size, memory_slots, memory_width)
        
        # Apply each write head sequentially
        for head in range(self.num_write_heads):
            # Extract weights and parameters for this head
            head_weights = write_weights[:, head].unsqueeze(2)  # (batch_size, memory_slots, 1)
            erase, add = write_params[head]
            
            # Reshape erase and add for broadcasting
            erase = erase.unsqueeze(1)  # (batch_size, 1, memory_width)
            add = add.unsqueeze(1)      # (batch_size, 1, memory_width)
            
            # Erase old content
            memory = memory * (1 - torch.bmm(head_weights, erase))
            
            # Add new content
            memory = memory + torch.bmm(head_weights, add)
            
        return memory
            
    def forward(self, x, prev_state=None):
        """
        Process input through the NTM
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            prev_state: Previous NTM state
            
        Returns:
            output: Output tensor
            state: Updated NTM state
        """
        batch_size = x.size(0)
        
        # Initialize state if not provided
        if prev_state is None:
            state = self.init_state(batch_size, x.device)
        else:
            state = prev_state
            
        # Unpack state
        memory = state['memory']
        read_weights = state['read_weights']
        write_weights = state['write_weights']
        read_vectors = state['read_vectors']
        controller_state = state['controller_state']
        
        # Prepare controller input (concatenate input with previous read vectors)
        controller_input = torch.cat([x, read_vectors], dim=1)
        
        # Run controller
        if isinstance(self.controller, nn.LSTMCell):
            hidden, cell = self.controller(controller_input, controller_state)
            controller_output = hidden
            controller_state = (hidden, cell)
        else:
            hidden = self.controller(controller_input, controller_state)
            controller_output = hidden
            controller_state = hidden
            
        # Update write weights and write to memory
        write_weights, write_params = self.write_heads.address_memory(
            controller_output, write_weights, memory, head_type='write'
        )
        memory = self.write_memory(memory, write_weights, write_params)
        
        # Update read weights and read from memory
        read_weights, _ = self.read_heads.address_memory(
            controller_output, read_weights, memory, head_type='read'
        )
        read_vectors = self.read_memory(memory, read_weights)
        
        # Compute output
        output = self.output_layer(
            torch.cat([controller_output, read_vectors], dim=1)
        )
        
        # Pack updated state
        state = {
            'memory': memory,
            'read_weights': read_weights,
            'write_weights': write_weights,
            'read_vectors': read_vectors,
            'controller_state': controller_state
        }
        
        return output, state 