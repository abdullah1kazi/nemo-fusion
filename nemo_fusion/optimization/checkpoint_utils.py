"""
Checkpoint Utilities for NeMo Fusion.

Provides efficient checkpoint saving, loading, and compression
for distributed training.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import json
import time


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000  # Save every N steps
    keep_last_n: int = 3  # Keep last N checkpoints
    async_save: bool = True  # Asynchronous checkpoint saving
    compression: bool = False  # Compress checkpoints
    
    # Distributed checkpointing
    use_distributed_checkpoint: bool = True
    shard_checkpoints: bool = True


class CheckpointManager:
    """
    Checkpoint manager for distributed training.
    
    Features:
    - Automatic checkpoint saving at intervals
    - Keep only last N checkpoints
    - Distributed checkpoint support
    - Async saving to avoid blocking training
    
    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="./checkpoints",
        ...     save_interval=1000,
        ...     keep_last_n=3
        ... )
        >>> 
        >>> # During training
        >>> if manager.should_save(step):
        ...     manager.save_checkpoint(
        ...         step=step,
        ...         model=model,
        ...         optimizer=optimizer,
        ...         metrics={"loss": loss.item()}
        ...     )
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_interval: int = 1000,
        keep_last_n: int = 3,
        async_save: bool = False,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N steps
            keep_last_n: Keep last N checkpoints (delete older ones)
            async_save: Use asynchronous saving (experimental)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.async_save = async_save
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints: List[Path] = []
        
        # Distributed info
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
    
    def should_save(self, step: int) -> bool:
        """
        Check if checkpoint should be saved at this step.
        
        Args:
            step: Current training step
        
        Returns:
            True if checkpoint should be saved
        """
        return step > 0 and step % self.save_interval == 0
    
    def save_ckpt(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler to save
            metrics: Training metrics to save
        
        Returns:
            Path to saved checkpoint
        """
        # Only rank 0 saves in distributed training
        if self.is_distributed and self.rank != 0:
            return None
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        # Prepare checkpoint data
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "timestamp": time.time(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track saved checkpoint
        self.saved_checkpoints.append(checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_old_ckpts()

        print(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_ckpt(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
        
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        metadata = {
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
            "timestamp": checkpoint.get("timestamp", 0),
        }
        
        print(f"Loaded checkpoint from {checkpoint_path} (step {metadata['step']})")
        
        return metadata
    
    def _cleanup_old_ckpts(self):
        """Remove old checkpoints, keeping only last N."""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # Sort by modification time
            self.saved_checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest checkpoints
            while len(self.saved_checkpoints) > self.keep_last_n:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"Removed old checkpoint: {old_checkpoint}")
    
    def get_latest_ckpt(self) -> Optional[Path]:
        """
        Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        return checkpoints[-1]


class CheckpointCompressor:
    """
    Checkpoint compressor for reducing storage size.
    
    Applies various compression techniques:
    - Quantization of weights
    - Sparse storage
    - Dictionary compression
    """
    
    @staticmethod
    def compress_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compress model state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            Compressed state dictionary
        """
        # Simple implementation: convert to half precision
        compressed = {}
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                compressed[key] = tensor.half()
            else:
                compressed[key] = tensor
        return compressed
    
    @staticmethod
    def decompress_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decompress model state dict.

        Args:
            state_dict: Compressed state dictionary

        Returns:
            Decompressed state dictionary
        """
        # Convert back to float32
        decompressed = {}
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float16:
                decompressed[key] = tensor.float()
            else:
                decompressed[key] = tensor
        return decompressed

