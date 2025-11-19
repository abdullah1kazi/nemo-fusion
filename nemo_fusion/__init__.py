"""
NeMo Fusion: Advanced Parallelism and Optimization Toolkit for NeMo Framework

A high-performance extension toolkit for NVIDIA NeMo Framework that provides:
- Advanced parallelism strategies and auto-optimization
- Distributed training profiling and bottleneck analysis  
- Memory-efficient attention mechanisms
- Multi-modal training extensions
"""

__version__ = "0.1.0"
__author__ = "NeMo Fusion Contributors"
__license__ = "Apache 2.0"

from nemo_fusion.parallelism import (
    AutoParallelOptimizer,
    HybridParallelStrategy,
    ParallelStrategy,
)
from nemo_fusion.profiling import DistributedProfiler, GPUProfiler
from nemo_fusion.optimization import MixedPrecisionConfig, GradientOptimizer

__all__ = [
    "__version__",
    "AutoParallelOptimizer",
    "HybridParallelStrategy",
    "ParallelStrategy",
    "DistributedProfiler",
    "GPUProfiler",
    "MixedPrecisionConfig",
    "GradientOptimizer",
]

