"""
Gradient Optimization for NeMo Fusion.

Provides gradient accumulation, clipping, and optimization strategies
for efficient distributed training.
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class GradientConfig:
    """Configuration for gradient optimization."""
    
    gradient_accumulation_steps: int = 1
    gradient_clip_val: Optional[float] = 1.0
    gradient_clip_algorithm: str = "norm"  # "norm" or "value"
    
    # Dynamic gradient accumulation
    use_dynamic_accumulation: bool = False
    min_accumulation_steps: int = 1
    max_accumulation_steps: int = 16
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False
    checkpoint_num_layers: int = 1


class GradientOptimizer:
    """
    Gradient optimizer with accumulation and clipping.
    
    Provides:
    - Gradient accumulation for larger effective batch sizes
    - Gradient clipping to prevent exploding gradients
    - Dynamic gradient accumulation based on memory usage
    
    Example:
        >>> grad_optimizer = GradientOptimizer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     gradient_accumulation_steps=4,
        ...     gradient_clip_val=1.0
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     grad_optimizer.backward(loss)
        ...     
        ...     if grad_optimizer.should_step():
        ...         grad_optimizer.step()
        ...         grad_optimizer.zero_grad()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = 1.0,
        gradient_clip_algorithm: str = "norm",
    ):
        """
        Initialize gradient optimizer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_clip_val: Gradient clipping value (None = no clipping)
            gradient_clip_algorithm: "norm" or "value"
        """
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        
        self.current_step = 0
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass with gradient accumulation.
        
        Args:
            loss: Loss tensor
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.current_step % self.gradient_accumulation_steps == 0
    
    def step(self):
        """Optimizer step with gradient clipping."""
        # Clip gradients if specified
        if self.gradient_clip_val is not None:
            if self.gradient_clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            elif self.gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
        
        # Optimizer step
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_grad_norm(self) -> float:
        """
        Calculate gradient norm across all parameters.
        
        Returns:
            Gradient norm value
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


class GradientAccumulationScheduler:
    """
    Dynamic gradient accumulation scheduler.
    
    Adjusts gradient accumulation steps based on:
    - Memory usage
    - Training progress
    - Custom schedules
    
    Example:
        >>> scheduler = GradientAccumulationScheduler(
        ...     initial_steps=4,
        ...     schedule_fn=lambda step: min(16, 4 * (step // 1000 + 1))
        ... )
        >>> 
        >>> for step, batch in enumerate(dataloader):
        ...     accumulation_steps = scheduler.get_accumulation_steps(step)
        ...     # Use accumulation_steps for training
    """
    
    def __init__(
        self,
        initial_steps: int = 1,
        schedule_fn: Optional[Callable[[int], int]] = None,
        min_steps: int = 1,
        max_steps: int = 16,
    ):
        """
        Initialize gradient accumulation scheduler.
        
        Args:
            initial_steps: Initial accumulation steps
            schedule_fn: Function that takes step and returns accumulation steps
            min_steps: Minimum accumulation steps
            max_steps: Maximum accumulation steps
        """
        self.initial_steps = initial_steps
        self.schedule_fn = schedule_fn
        self.min_steps = min_steps
        self.max_steps = max_steps
        
        self.current_steps = initial_steps
    
    def get_accum_steps(self, step: int) -> int:
        """
        Get accumulation steps for current training step.

        Args:
            step: Current training step

        Returns:
            Number of accumulation steps
        """
        if self.schedule_fn is not None:
            steps = self.schedule_fn(step)
        else:
            steps = self.initial_steps
        
        # Clamp to min/max
        steps = max(self.min_steps, min(self.max_steps, steps))
        self.current_steps = steps
        
        return steps

