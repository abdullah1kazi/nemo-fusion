from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import torch

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUMetrics:
    """GPU performance metrics at a point in time."""
    
    timestamp: float
    gpu_id: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    
    @property
    def memory_free_mb(self) -> float:
        """Calculate free memory in MB."""
        return self.memory_total_mb - self.memory_used_mb


@dataclass
class GPUProfileReport:
    """Aggregated GPU profiling report."""
    
    gpu_id: int
    num_samples: int
    avg_utilization: float
    max_utilization: float
    min_utilization: float
    avg_memory_used_mb: float
    max_memory_used_mb: float
    peak_memory_percent: float
    avg_temperature_c: Optional[float] = None
    max_temperature_c: Optional[float] = None
    
    def __str__(self) -> str:
        report = [
            f"GPU {self.gpu_id} Profile Report:",
            f"  Samples: {self.num_samples}",
            f"  Utilization: avg={self.avg_utilization:.1f}%, "
            f"max={self.max_utilization:.1f}%, min={self.min_utilization:.1f}%",
            f"  Memory: avg={self.avg_memory_used_mb:.0f}MB, "
            f"max={self.max_memory_used_mb:.0f}MB, peak={self.peak_memory_percent:.1f}%",
        ]
        
        if self.avg_temperature_c is not None:
            report.append(
                f"  Temperature: avg={self.avg_temperature_c:.1f}°C, "
                f"max={self.max_temperature_c:.1f}°C"
            )
        
        return "\n".join(report)


class GPUProfiler:
    """
    GPU profiler for monitoring GPU performance during training.
    
    Tracks:
    - GPU utilization
    - Memory usage
    - Temperature
    - Power consumption
    - Clock speeds
    
    Example:
        >>> profiler = GPUProfiler()
        >>> profiler.start()
        >>> # ... training code ...
        >>> profiler.stop()
        >>> report = profiler.get_report()
        >>> print(report)
    """
    
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        sample_interval: float = 0.1,
    ):
        """
        Initialize GPU profiler.
        
        Args:
            gpu_ids: List of GPU IDs to monitor (None = all available)
            sample_interval: Sampling interval in seconds
        """
        self.sample_interval = sample_interval
        self.is_profiling = False
        self.metrics: Dict[int, List[GPUMetrics]] = {}
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
        
        # Determine GPU IDs to monitor
        if gpu_ids is None:
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = []
        else:
            self.gpu_ids = gpu_ids
        
        # Initialize metrics storage
        for gpu_id in self.gpu_ids:
            self.metrics[gpu_id] = []
    
    def start(self):
        """Start profiling."""
        self.is_profiling = True
        self.start_time = time.time()
    
    def stop(self):
        """Stop profiling."""
        self.is_profiling = False
        self.end_time = time.time()
    
    def sample(self):
        """Take a single sample of GPU metrics."""
        if not self.is_profiling:
            return
        
        timestamp = time.time()
        
        for gpu_id in self.gpu_ids:
            metrics = self._sample_gpu(gpu_id, timestamp)
            if metrics:
                self.metrics[gpu_id].append(metrics)
    
    def _sample_gpu(self, gpu_id: int, timestamp: float) -> Optional[GPUMetrics]:
        """Sample metrics for a single GPU."""
        try:
            # Get PyTorch memory stats
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2  # MB
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
            else:
                memory_allocated = 0
                memory_total = 0
            
            # Get NVML metrics if available
            utilization = 0.0
            temperature = None
            power_usage = None
            sm_clock = None
            
            if self.nvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = float(util.gpu)
                    
                    try:
                        temperature = float(pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ))
                    except:
                        pass
                    
                    try:
                        power_usage = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # W
                    except:
                        pass
                    
                    try:
                        sm_clock = float(pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_SM
                        ))
                    except:
                        pass
                except Exception as e:
                    pass
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_id=gpu_id,
                utilization_percent=utilization,
                memory_used_mb=memory_allocated,
                memory_total_mb=memory_total,
                memory_percent=(memory_allocated / memory_total * 100) if memory_total > 0 else 0,
                temperature_c=temperature,
                power_usage_w=power_usage,
                sm_clock_mhz=sm_clock,
            )
        
        except Exception as e:
            print(f"Error sampling GPU {gpu_id}: {e}")
            return None

    def get_report(self, gpu_id: Optional[int] = None) -> Dict[int, GPUProfileReport]:
        """
        Generate profiling report.

        Args:
            gpu_id: Specific GPU ID to report (None = all GPUs)

        Returns:
            Dictionary mapping GPU ID to GPUProfileReport
        """
        reports = {}

        gpu_ids_to_report = [gpu_id] if gpu_id is not None else self.gpu_ids

        for gid in gpu_ids_to_report:
            if gid not in self.metrics or not self.metrics[gid]:
                continue

            metrics_list = self.metrics[gid]

            # Calculate statistics
            utilizations = [m.utilization_percent for m in metrics_list]
            memory_used = [m.memory_used_mb for m in metrics_list]
            memory_percents = [m.memory_percent for m in metrics_list]
            temperatures = [m.temperature_c for m in metrics_list if m.temperature_c is not None]

            report = GPUProfileReport(
                gpu_id=gid,
                num_samples=len(metrics_list),
                avg_utilization=sum(utilizations) / len(utilizations),
                max_utilization=max(utilizations),
                min_utilization=min(utilizations),
                avg_memory_used_mb=sum(memory_used) / len(memory_used),
                max_memory_used_mb=max(memory_used),
                peak_memory_percent=max(memory_percents),
                avg_temperature_c=sum(temperatures) / len(temperatures) if temperatures else None,
                max_temperature_c=max(temperatures) if temperatures else None,
            )

            reports[gid] = report

        return reports

    def print_report(self):
        """Print profiling report for all GPUs."""
        reports = self.get_report()

        print("\n" + "="*60)
        print("GPU Profiling Report")
        print("="*60)

        for gpu_id, report in sorted(reports.items()):
            print(f"\n{report}")

        print("="*60 + "\n")

    def reset(self):
        """Reset all collected metrics."""
        for gpu_id in self.gpu_ids:
            self.metrics[gpu_id] = []

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
