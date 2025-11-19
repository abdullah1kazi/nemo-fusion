"""
Mixed Precision Training for NeMo Fusion.

Provides configurations and utilities for mixed precision training
with FP8, FP16, and BF16 on NVIDIA GPUs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import torch


class PrecisionType(Enum):
    """Supported precision types."""
    
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    
    @property
    def bytes_per_param(self) -> int:
        """Get bytes per parameter for this precision."""
        if self == PrecisionType.FP32:
            return 4
        elif self in (PrecisionType.FP16, PrecisionType.BF16):
            return 2
        elif self == PrecisionType.FP8:
            return 1
        return 4


@dataclass
class MixedPrecisionConfig:
    """
    Configuration for mixed precision training.
    
    Attributes:
        precision: Primary precision type (fp16, bf16, fp8)
        master_weights_dtype: Dtype for master weights (typically fp32)
        gradient_dtype: Dtype for gradients
        use_loss_scaling: Whether to use loss scaling (for fp16)
        initial_loss_scale: Initial loss scale value
        loss_scale_window: Window for loss scale adjustment
        hysteresis: Hysteresis for loss scale adjustment
        min_loss_scale: Minimum loss scale value
        fp8_margin: Margin for FP8 scaling
        fp8_interval: Interval for FP8 amax history update
        fp8_amax_history_len: Length of FP8 amax history
    """
    
    precision: PrecisionType = PrecisionType.BF16
    master_weights_dtype: torch.dtype = torch.float32
    gradient_dtype: Optional[torch.dtype] = None
    
    # Loss scaling (for FP16)
    use_loss_scaling: bool = True
    initial_loss_scale: float = 2**16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0
    
    # FP8 specific
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "most_recent"
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.gradient_dtype is None:
            if self.precision == PrecisionType.FP16:
                self.gradient_dtype = torch.float16
            elif self.precision == PrecisionType.BF16:
                self.gradient_dtype = torch.bfloat16
            elif self.precision == PrecisionType.FP8:
                self.gradient_dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
            else:
                self.gradient_dtype = torch.float32
        
        # FP16 requires loss scaling
        if self.precision == PrecisionType.FP16:
            self.use_loss_scaling = True
        
        # BF16 typically doesn't need loss scaling
        if self.precision == PrecisionType.BF16:
            self.use_loss_scaling = False
    
    def to_nemo_config(self) -> Dict[str, Any]:
        """
        Convert to NeMo Framework configuration format.
        
        Returns:
            Dictionary compatible with NeMo's precision configuration
        """
        config = {
            "precision": self.precision.value,
        }
        
        if self.use_loss_scaling:
            config["loss_scaling"] = {
                "initial_scale": self.initial_loss_scale,
                "scale_window": self.loss_scale_window,
                "hysteresis": self.hysteresis,
                "min_scale": self.min_loss_scale,
            }
        
        if self.precision == PrecisionType.FP8:
            config["fp8"] = {
                "margin": self.fp8_margin,
                "interval": self.fp8_interval,
                "amax_history_len": self.fp8_amax_history_len,
                "amax_compute_algo": self.fp8_amax_compute_algo,
            }
        
        return config
    
    @classmethod
    def for_h100(cls, use_fp8: bool = True) -> "MixedPrecisionConfig":
        """
        Optimized configuration for H100 GPUs.
        
        Args:
            use_fp8: Whether to use FP8 (H100 supports FP8)
        
        Returns:
            MixedPrecisionConfig optimized for H100
        """
        if use_fp8:
            return cls(
                precision=PrecisionType.FP8,
                fp8_margin=0,
                fp8_interval=1,
                fp8_amax_history_len=1024,
            )
        else:
            return cls(
                precision=PrecisionType.BF16,
                use_loss_scaling=False,
            )
    
    @classmethod
    def for_a100(cls) -> "MixedPrecisionConfig":
        """
        Optimized configuration for A100 GPUs.
        
        A100 doesn't support FP8, so use BF16.
        
        Returns:
            MixedPrecisionConfig optimized for A100
        """
        return cls(
            precision=PrecisionType.BF16,
            use_loss_scaling=False,
        )
    
    @classmethod
    def for_v100(cls) -> "MixedPrecisionConfig":
        """
        Optimized configuration for V100 GPUs.
        
        V100 doesn't support BF16, so use FP16 with loss scaling.
        
        Returns:
            MixedPrecisionConfig optimized for V100
        """
        return cls(
            precision=PrecisionType.FP16,
            use_loss_scaling=True,
            initial_loss_scale=2**16,
        )

