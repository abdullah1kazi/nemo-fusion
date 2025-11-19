"""
Optimization module for NeMo Fusion.

Provides training optimization utilities:
- Mixed precision training configurations
- Gradient optimization strategies
- Checkpoint compression
"""

from nemo_fusion.optimization.mixed_precision import (
    MixedPrecisionConfig,
    PrecisionType,
)
from nemo_fusion.optimization.gradient_optimization import (
    GradientOptimizer,
    GradientAccumulationScheduler,
)
from nemo_fusion.optimization.checkpoint_utils import (
    CheckpointCompressor,
    CheckpointManager,
)

__all__ = [
    "MixedPrecisionConfig",
    "PrecisionType",
    "GradientOptimizer",
    "GradientAccumulationScheduler",
    "CheckpointCompressor",
    "CheckpointManager",
]

