"""
Hybrid Parallelism Strategies for NeMo Fusion.

Provides advanced hybrid parallelism configurations that combine
multiple parallelism types for optimal performance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import math


class ParallelismType(Enum):
    """Types of parallelism supported."""
    
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"
    CONTEXT = "context"
    EXPERT = "expert"  # For MoE models
    SEQUENCE = "sequence"


@dataclass
class HybridParallelStrategy:
    """
    Hybrid parallelism configuration combining multiple strategies.
    
    This class provides pre-configured hybrid strategies optimized
    for different model sizes and hardware configurations.
    """
    
    name: str
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    context_parallel: int = 1
    expert_parallel: int = 1
    sequence_parallel: bool = False
    virtual_pipeline_parallel: Optional[int] = None
    
    def __str__(self) -> str:
        config = [
            f"Hybrid Strategy: {self.name}",
            f"  TP={self.tensor_parallel}",
            f"  PP={self.pipeline_parallel}",
            f"  DP={self.data_parallel}",
        ]
        
        if self.context_parallel > 1:
            config.append(f"  CP={self.context_parallel}")
        if self.expert_parallel > 1:
            config.append(f"  EP={self.expert_parallel}")
        if self.sequence_parallel:
            config.append(f"  SP=enabled")
        if self.virtual_pipeline_parallel:
            config.append(f"  VPP={self.virtual_pipeline_parallel}")
        
        return "\n".join(config)
    
    def to_nemo_config(self) -> Dict:
        """Convert to NeMo Framework configuration."""
        config = {
            "tensor_model_parallel_size": self.tensor_parallel,
            "pipeline_model_parallel_size": self.pipeline_parallel,
            "data_parallel_size": self.data_parallel,
            "context_parallel_size": self.context_parallel,
            "sequence_parallel": self.sequence_parallel,
        }
        
        if self.expert_parallel > 1:
            config["expert_model_parallel_size"] = self.expert_parallel
        
        if self.virtual_pipeline_parallel:
            config["virtual_pipeline_model_parallel_size"] = self.virtual_pipeline_parallel
        
        return config
    
    @property
    def total_gpus(self) -> int:
        """Calculate total number of GPUs required."""
        return (
            self.tensor_parallel *
            self.pipeline_parallel *
            self.data_parallel *
            self.context_parallel *
            self.expert_parallel
        )
    
    @classmethod
    def for_small_model(cls, num_gpus: int = 8) -> "HybridParallelStrategy":
        """
        Strategy for small models (< 10B parameters).
        
        Prioritizes data parallelism for maximum throughput.
        """
        return cls(
            name="Small Model (< 10B)",
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=num_gpus,
            sequence_parallel=False,
        )
    
    @classmethod
    def for_medium_model(cls, num_gpus: int = 8) -> "HybridParallelStrategy":
        """
        Strategy for medium models (10B - 70B parameters).
        
        Balances tensor and data parallelism.
        """
        tp = min(4, num_gpus)
        dp = num_gpus // tp
        
        return cls(
            name="Medium Model (10B-70B)",
            tensor_parallel=tp,
            pipeline_parallel=1,
            data_parallel=dp,
            sequence_parallel=True,  # Enable with TP
        )
    
    @classmethod
    def for_large_model(cls, num_gpus: int = 16) -> "HybridParallelStrategy":
        """
        Strategy for large models (70B - 175B parameters).
        
        Uses hybrid TP + PP + DP.
        """
        tp = min(8, num_gpus // 2)
        pp = 2
        dp = num_gpus // (tp * pp)
        
        return cls(
            name="Large Model (70B-175B)",
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            sequence_parallel=True,
            virtual_pipeline_parallel=2,  # Reduce pipeline bubble
        )
    
    @classmethod
    def for_very_large_model(cls, num_gpus: int = 64) -> "HybridParallelStrategy":
        """
        Strategy for very large models (> 175B parameters).
        
        Uses aggressive hybrid parallelism with all strategies.
        """
        tp = 8
        pp = 4
        dp = num_gpus // (tp * pp)
        
        return cls(
            name="Very Large Model (> 175B)",
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            context_parallel=2,  # For long sequences
            sequence_parallel=True,
            virtual_pipeline_parallel=4,
        )

