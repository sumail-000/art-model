import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Callable
import higher
import copy

class OnlineMetaLearning:
    """
    Online Meta-Learning (OML) for model adaptation in real-time
    during user interactions.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.001,
                 meta_lr: float = 0.0001,
                 first_order: bool = True,
                 retain_graph: bool = False):
        """
        Initialize Online Meta-Learning
        
        Args:
            model: PyTorch model to be adapted
            inner_lr: Learning rate for inner adaptation
            meta_lr: Learning rate for meta update
            first_order: Whether to use first-order approximation
            retain_graph: Whether to retain computation graph during optimization
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.first_order = first_order
        self.retain_graph = retain_graph
        
        # Create meta optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Keep track of adaptation state
        self.adaptation_steps = 0
        self.adaptation_history = []
        
        # Tasks/users seen so far (for potential task/user clustering)
        self.tasks_seen = set()
        
    def adapt(self, 
              support_data: Dict[str, torch.Tensor], 
              loss_fn: Callable, 
              steps: int = 1,
              task_id: Optional[str] = None) -> nn.Module:
        """
        Adapt the model to new data using gradient descent
        
        Args:
            support_data: Dictionary of tensors for adaptation
            loss_fn: Loss function for inner adaptation
            steps: Number of adaptation steps
            task_id: Optional identifier for this adaptation task/user
            
        Returns:
            Adapted model copy
        """
        if task_id is not None:
            self.tasks_seen.add(task_id)
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        
        # Set to training mode
        adapted_model.train()
        
        # Create optimizer for inner loop
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform adaptation steps
        for step in range(steps):
            # Forward pass
            outputs = adapted_model(**support_data)
            
            # Compute loss
            loss = loss_fn(outputs, support_data)
            
            # Backward pass
            inner_optimizer.zero_grad()
            loss.backward(retain_graph=self.retain_graph)
            inner_optimizer.step()
            
            # Track adaptation
            self.adaptation_steps += 1
            self.adaptation_history.append({
                'task_id': task_id,
                'step': self.adaptation_steps,
                'loss': loss.item()
            })
        
        # Return adapted model
        return adapted_model
    
    def meta_update(self, 
                   support_data: Dict[str, torch.Tensor],
                   query_data: Dict[str, torch.Tensor],
                   loss_fn: Callable,
                   inner_steps: int = 1,
                   task_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Perform a meta-update using MAML-style optimization
        
        Args:
            support_data: Data for inner loop adaptation
            query_data: Data for outer loop evaluation
            loss_fn: Loss function
            inner_steps: Number of inner adaptation steps
            task_id: Optional identifier for this adaptation task/user
            
        Returns:
            Tuple of (pre-adaptation loss, post-adaptation loss)
        """
        if task_id is not None:
            self.tasks_seen.add(task_id)
            
        # Set model to training
        self.model.train()
        
        # Evaluate pre-adaptation performance
        with torch.no_grad():
            pre_outputs = self.model(**query_data)
            pre_loss = loss_fn(pre_outputs, query_data).item()
        
        # Higher-order meta-learning (use higher library)
        inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Create stateless version of model and optimizer for differentiable optimization
        with higher.innerloop_ctx(self.model, inner_opt, 
                                  copy_initial_weights=False,
                                  track_higher_grads=not self.first_order) as (fmodel, diffopt):
            # Inner loop adaptation
            for _ in range(inner_steps):
                outputs = fmodel(**support_data)
                inner_loss = loss_fn(outputs, support_data)
                diffopt.step(inner_loss)
            
            # Outer loop evaluation on query set
            query_outputs = fmodel(**query_data)
            query_loss = loss_fn(query_outputs, query_data)
            
            # Meta optimization
            self.meta_optimizer.zero_grad()
            query_loss.backward()
            self.meta_optimizer.step()
            
            # Track adaptation
            self.adaptation_steps += 1
            self.adaptation_history.append({
                'task_id': task_id,
                'step': self.adaptation_steps,
                'meta_loss': query_loss.item()
            })
        
        # Set model back to eval mode
        self.model.eval()
        
        # Return losses for monitoring
        return pre_loss, query_loss.item()
    
    def reset_adaptation(self):
        """Reset adaptation history"""
        self.adaptation_steps = 0
        self.adaptation_history = []
    
    def save_state(self, path: str):
        """Save adaptation state"""
        torch.save({
            'adaptation_steps': self.adaptation_steps,
            'adaptation_history': self.adaptation_history,
            'tasks_seen': list(self.tasks_seen),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.meta_optimizer.state_dict()
        }, path)
    
    def load_state(self, path: str):
        """Load adaptation state"""
        state = torch.load(path)
        self.adaptation_steps = state['adaptation_steps']
        self.adaptation_history = state['adaptation_history']
        self.tasks_seen = set(state['tasks_seen'])
        self.model.load_state_dict(state['model_state'])
        self.meta_optimizer.load_state_dict(state['optimizer_state'])


class MAMLWrapper(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) wrapper for adapting modules 
    to improve online learning efficiency
    """
    
    def __init__(self, module: nn.Module):
        """
        Wrap a PyTorch module for meta-learning
        
        Args:
            module: PyTorch module to wrap
        """
        super().__init__()
        self.module = module
        self.meta_parameters = None
        
    def forward(self, *args, **kwargs):
        """Forward pass using current parameters"""
        return self.module(*args, **kwargs)
    
    def clone(self):
        """Create a copy with shared meta parameters but separate fast parameters"""
        clone = MAMLWrapper(copy.deepcopy(self.module))
        clone.meta_parameters = self.meta_parameters
        return clone
    
    def adapt(self, loss: torch.Tensor, lr: float = 0.01):
        """Adapt parameters based on loss"""
        grads = torch.autograd.grad(loss, self.module.parameters(), 
                                    create_graph=True, retain_graph=True)
        
        # Store meta parameters if not already saved
        if self.meta_parameters is None:
            self.meta_parameters = [p.clone() for p in self.module.parameters()]
        
        # Update module parameters in-place
        for p, g, mp in zip(self.module.parameters(), grads, self.meta_parameters):
            p.data = mp - lr * g
    
    def reset(self):
        """Reset to meta parameters"""
        if self.meta_parameters is not None:
            for p, mp in zip(self.module.parameters(), self.meta_parameters):
                p.data = mp.clone()


class OnlineMetaLearningScheduler:
    """
    Scheduler to control when to perform meta-updates
    and how to prioritize different adaptation tasks
    """
    
    def __init__(self, 
                 oml: OnlineMetaLearning,
                 update_frequency: int = 10,
                 prioritize_recent: bool = True,
                 task_importance_fn: Optional[Callable] = None):
        """
        Initialize scheduler
        
        Args:
            oml: OnlineMetaLearning instance
            update_frequency: How often to perform meta-updates
            prioritize_recent: Whether to give higher weights to recent tasks
            task_importance_fn: Optional function to assign importance to tasks
        """
        self.oml = oml
        self.update_frequency = update_frequency
        self.prioritize_recent = prioritize_recent
        self.task_importance_fn = task_importance_fn
        
        # Task buffer for replay
        self.task_buffer = []
        self.max_buffer_size = 100
        
    def step(self, 
             support_data: Dict[str, torch.Tensor],
             query_data: Dict[str, torch.Tensor],
             loss_fn: Callable,
             task_id: Optional[str] = None,
             importance: float = 1.0):
        """
        Process a new task and decide whether to perform a meta-update
        
        Args:
            support_data: Support data for adaptation
            query_data: Query data for meta-update
            loss_fn: Loss function
            task_id: Optional task identifier
            importance: Task importance (if not using task_importance_fn)
            
        Returns:
            Dictionary with update information
        """
        # Add to task buffer
        self.task_buffer.append({
            'support_data': support_data,
            'query_data': query_data,
            'task_id': task_id,
            'timestamp': self.oml.adaptation_steps,
            'importance': importance if self.task_importance_fn is None 
                          else self.task_importance_fn(task_id, support_data, query_data)
        })
        
        # Manage buffer size
        if len(self.task_buffer) > self.max_buffer_size:
            # Remove least important task
            if self.prioritize_recent:
                # Sort by timestamp, oldest first
                self.task_buffer.sort(key=lambda x: x['timestamp'])
            else:
                # Sort by importance, least important first
                self.task_buffer.sort(key=lambda x: x['importance'])
                
            # Remove first item (least important/oldest)
            self.task_buffer.pop(0)
        
        # Decide whether to perform meta-update
        perform_update = (self.oml.adaptation_steps % self.update_frequency) == 0
        
        result = {
            'performed_update': False,
            'pre_loss': None,
            'post_loss': None
        }
        
        if perform_update:
            # Simple case: just use the current task
            pre_loss, post_loss = self.oml.meta_update(
                support_data, query_data, loss_fn, task_id=task_id
            )
            
            result['performed_update'] = True
            result['pre_loss'] = pre_loss
            result['post_loss'] = post_loss
            
        return result
    
    def force_update(self, loss_fn: Callable, num_tasks: int = 5):
        """
        Force a meta-update using tasks from the buffer
        
        Args:
            loss_fn: Loss function
            num_tasks: Number of tasks to sample for update
            
        Returns:
            Dictionary with update metrics
        """
        if not self.task_buffer:
            return {'performed_update': False}
        
        # Select tasks for update
        if self.prioritize_recent:
            # Sort by timestamp, most recent first
            sorted_tasks = sorted(self.task_buffer, 
                                 key=lambda x: x['timestamp'], 
                                 reverse=True)
        else:
            # Sort by importance, most important first
            sorted_tasks = sorted(self.task_buffer, 
                                 key=lambda x: x['importance'], 
                                 reverse=True)
        
        # Take top N tasks
        selected_tasks = sorted_tasks[:min(num_tasks, len(sorted_tasks))]
        
        # Perform meta-update for each task
        pre_losses = []
        post_losses = []
        
        for task in selected_tasks:
            pre_loss, post_loss = self.oml.meta_update(
                task['support_data'], 
                task['query_data'], 
                loss_fn,
                task_id=task['task_id']
            )
            pre_losses.append(pre_loss)
            post_losses.append(post_loss)
        
        return {
            'performed_update': True,
            'num_tasks': len(selected_tasks),
            'pre_losses': pre_losses,
            'post_losses': post_losses,
            'avg_pre_loss': sum(pre_losses) / len(pre_losses) if pre_losses else None,
            'avg_post_loss': sum(post_losses) / len(post_losses) if post_losses else None
        } 