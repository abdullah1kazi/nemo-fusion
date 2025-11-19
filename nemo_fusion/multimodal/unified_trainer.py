"""
Unified Multi-Modal Trainer for NeMo Fusion.

Provides a unified training interface for multi-modal models
with support for distributed training and NeMo integration.
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.distributed as dist


@dataclass
class TrainingConfig:
    """Configuration for multi-modal training."""
    
    # Training hyperparameters
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Optimization
    gradient_accumulation_steps: int = 1
    gradient_clip_val: Optional[float] = 1.0
    mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    
    # Distributed training
    use_distributed: bool = False
    local_rank: int = 0


class UnifiedMultiModalTrainer:
    """
    Unified trainer for multi-modal models.
    
    Provides a high-level training interface with support for:
    - Multi-modal data loading
    - Distributed training
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Logging and evaluation
    
    Example:
        >>> model = MultiModalModel(...)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> 
        >>> trainer = UnifiedMultiModalTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     config=TrainingConfig(num_epochs=10)
        ... )
        >>> 
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
    ):
        """
        Initialize unified trainer.
        
        Args:
            model: Multi-modal model
            optimizer: Optimizer
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            loss_fn: Custom loss function
            metrics_fn: Custom metrics function
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics_fn = metrics_fn
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Distributed setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.config.local_rank]
            )
        else:
            self.rank = 0
            self.world_size = 1
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
    
    def train(self):
        """Run training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()
            
            # Validation
            if self.val_dataloader is not None:
                val_loss = self.validate()
                
                if self.rank == 0:
                    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")
            
            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        print("Training completed!")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.config.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0 and self.rank == 0:
                    print(f"Step {self.global_step}: loss={loss.item():.4f}")
                
                # Checkpointing
                if self.global_step % self.config.checkpoint_interval == 0 and self.rank == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(batch)
                    loss = self.compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def compute_loss(self, outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss."""
        if "label" in batch:
            return self.loss_fn(outputs, batch["label"])
        return outputs  # Assume model returns loss directly
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device = next(self.model.parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/{filename}")
        print(f"Saved checkpoint: {filename}")

