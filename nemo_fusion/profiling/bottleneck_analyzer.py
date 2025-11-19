"""
Bottleneck Analyzer for NeMo Fusion.

Identifies performance bottlenecks in distributed training including:
- GPU utilization issues
- Communication overhead
- Memory bottlenecks
- Load imbalance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time
import torch
import torch.distributed as dist


class BottleneckType(Enum):
    """Types of bottlenecks that can be detected."""
    
    LOW_GPU_UTILIZATION = "low_gpu_utilization"
    HIGH_COMMUNICATION_OVERHEAD = "high_communication_overhead"
    MEMORY_BOTTLENECK = "memory_bottleneck"
    LOAD_IMBALANCE = "load_imbalance"
    PIPELINE_BUBBLE = "pipeline_bubble"
    IO_BOTTLENECK = "io_bottleneck"


@dataclass
class Bottleneck:
    """Represents a detected bottleneck."""
    
    type: BottleneckType
    severity: float  # 0.0 to 1.0
    description: str
    recommendation: str
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        severity_str = "HIGH" if self.severity > 0.7 else "MEDIUM" if self.severity > 0.4 else "LOW"
        return (
            f"[{severity_str}] {self.type.value}:\n"
            f"  Description: {self.description}\n"
            f"  Recommendation: {self.recommendation}\n"
            f"  Metrics: {self.metrics}"
        )


@dataclass
class BottleneckReport:
    """Comprehensive bottleneck analysis report."""
    
    bottlenecks: List[Bottleneck]
    overall_efficiency: float
    total_time_seconds: float
    gpu_utilization_avg: float
    communication_time_percent: float
    
    def __str__(self) -> str:
        report = [
            "\n" + "="*70,
            "Distributed Training Bottleneck Analysis",
            "="*70,
            f"Overall Efficiency: {self.overall_efficiency:.1%}",
            f"Total Time: {self.total_time_seconds:.2f}s",
            f"Avg GPU Utilization: {self.gpu_utilization_avg:.1%}",
            f"Communication Overhead: {self.communication_time_percent:.1%}",
            "",
            f"Detected {len(self.bottlenecks)} bottleneck(s):",
            ""
        ]
        
        for i, bottleneck in enumerate(self.bottlenecks, 1):
            report.append(f"{i}. {bottleneck}")
            report.append("")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    @property
    def has_critical(self) -> bool:
        """Check if there are any critical (high severity) bottlenecks."""
        return any(b.severity > 0.7 for b in self.bottlenecks)


class DistributedProfiler:
    """
    Distributed training profiler that identifies bottlenecks.
    
    Monitors:
    - GPU utilization across all ranks
    - Communication time
    - Memory usage
    - Load balance
    - Pipeline efficiency
    
    Example:
        >>> profiler = DistributedProfiler()
        >>> with profiler.profile():
        ...     # Training loop
        ...     for batch in dataloader:
        ...         output = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        >>> report = profiler.analyze()
        >>> print(report)
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """
        Initialize distributed profiler.
        
        Args:
            enable_detailed_profiling: Enable detailed per-operation profiling
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.is_profiling = False
        
        # Timing metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.compute_times: List[float] = []
        self.communication_times: List[float] = []
        self.io_times: List[float] = []
        
        # GPU metrics
        self.gpu_utilizations: List[float] = []
        self.memory_usages: List[float] = []
        
        # Distributed info
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
    
    def profile(self):
        """Context manager for profiling."""
        return self
    
    def __enter__(self):
        """Start profiling."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling."""
        self.stop()
        return False
    
    def start(self):
        """Start profiling."""
        self.is_profiling = True
        self.start_time = time.time()
        
        if self.enable_detailed_profiling and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def stop(self):
        """Stop profiling."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.end_time = time.time()
        self.is_profiling = False

    def record_comp_time(self, duration: float):
        """Record computation time."""
        self.compute_times.append(duration)

    def record_comm_time(self, duration: float):
        """Record communication time."""
        self.communication_times.append(duration)

    def record_io_time(self, duration: float):
        """Record I/O time."""
        self.io_times.append(duration)

    def record_gpu_util(self, utilization: float):
        """Record GPU utilization percentage."""
        self.gpu_utilizations.append(utilization)

    def record_mem_usage(self, usage_percent: float):
        """Record memory usage percentage."""
        self.memory_usages.append(usage_percent)

    def analyze(self) -> BottleneckReport:
        """
        Analyze collected metrics and identify bottlenecks.

        Returns:
            BottleneckReport with detected bottlenecks and recommendations
        """
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Profiler was not run. Call start() and stop() first.")

        total_time = self.end_time - self.start_time
        bottlenecks = []

        # Calculate metrics
        avg_gpu_util = sum(self.gpu_utilizations) / len(self.gpu_utilizations) if self.gpu_utilizations else 0.0
        total_comm_time = sum(self.communication_times)
        comm_percent = (total_comm_time / total_time * 100) if total_time > 0 else 0.0

        # Detect low GPU utilization
        if avg_gpu_util < 0.7:  # Less than 70% utilization
            severity = 1.0 - avg_gpu_util
            bottlenecks.append(Bottleneck(
                type=BottleneckType.LOW_GPU_UTILIZATION,
                severity=severity,
                description=f"Average GPU utilization is only {avg_gpu_util:.1%}",
                recommendation=(
                    "Consider: 1) Increasing batch size, "
                    "2) Reducing data loading overhead, "
                    "3) Optimizing data preprocessing"
                ),
                metrics={"avg_utilization": avg_gpu_util}
            ))

        # Detect high communication overhead
        if comm_percent > 20:  # More than 20% time in communication
            severity = min(1.0, comm_percent / 50)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.HIGH_COMMUNICATION_OVERHEAD,
                severity=severity,
                description=f"Communication takes {comm_percent:.1f}% of total time",
                recommendation=(
                    "Consider: 1) Reducing tensor/pipeline parallelism, "
                    "2) Increasing data parallelism, "
                    "3) Using gradient accumulation, "
                    "4) Enabling communication overlap"
                ),
                metrics={"communication_percent": comm_percent}
            ))

        # Detect memory bottleneck
        if self.memory_usages:
            avg_memory = sum(self.memory_usages) / len(self.memory_usages)
            if avg_memory > 0.9:  # More than 90% memory usage
                severity = (avg_memory - 0.8) / 0.2  # Scale from 80-100%
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.MEMORY_BOTTLENECK,
                    severity=severity,
                    description=f"Memory usage is at {avg_memory:.1%}",
                    recommendation=(
                        "Consider: 1) Reducing batch size, "
                        "2) Enabling activation checkpointing, "
                        "3) Increasing model parallelism, "
                        "4) Using mixed precision training"
                    ),
                    metrics={"avg_memory_usage": avg_memory}
                ))

        # Calculate overall efficiency
        efficiency = avg_gpu_util * (1.0 - comm_percent / 100)

        # Sort bottlenecks by severity
        bottlenecks.sort(key=lambda b: b.severity, reverse=True)

        return BottleneckReport(
            bottlenecks=bottlenecks,
            overall_efficiency=efficiency,
            total_time_seconds=total_time,
            gpu_utilization_avg=avg_gpu_util,
            communication_time_percent=comm_percent,
        )

    def get_recommends(self) -> List[str]:
        """
        Get optimization recommendations based on analysis.

        Returns:
            List of recommendation strings
        """
        report = self.analyze()

        if not report.bottlenecks:
            return ["No significant bottlenecks detected. Training is well optimized!"]

        recommendations = []
        for bottleneck in report.bottlenecks:
            if bottleneck.severity > 0.5:  # Only show medium+ severity
                recommendations.append(f"[{bottleneck.type.value}] {bottleneck.recommendation}")

        return recommendations
