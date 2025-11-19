"""
Communication Profiler for NeMo Fusion.

Profiles communication operations in distributed training including:
- All-reduce operations
- All-gather operations
- Point-to-point communication
- Communication bandwidth utilization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time
import torch
import torch.distributed as dist


class CommType(Enum):
    """Types of communication operations."""
    
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"


@dataclass
class CommEvent:
    """Represents a single communication event."""
    
    comm_type: CommType
    start_time: float
    end_time: float
    data_size_bytes: int
    rank: int
    group_size: int
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def bandwidth_gbps(self) -> float:
        """Effective bandwidth in GB/s."""
        if self.duration > 0:
            return (self.data_size_bytes / 1e9) / self.duration
        return 0.0


@dataclass
class CommProfile:
    """Communication profiling statistics."""
    
    comm_type: CommType
    num_calls: int
    total_time: float
    total_data_bytes: int
    avg_time: float
    avg_bandwidth_gbps: float
    
    def __str__(self) -> str:
        return (
            f"{self.comm_type.value}:\n"
            f"  Calls: {self.num_calls}\n"
            f"  Total time: {self.total_time:.3f}s\n"
            f"  Total data: {self.total_data_bytes / 1e9:.2f} GB\n"
            f"  Avg time: {self.avg_time*1000:.2f}ms\n"
            f"  Avg bandwidth: {self.avg_bandwidth_gbps:.2f} GB/s"
        )


class CommunicationProfiler:
    """
    Profiler for distributed communication operations.
    
    Tracks all communication operations and provides detailed statistics
    about communication patterns, bandwidth utilization, and overhead.
    
    Example:
        >>> profiler = CommunicationProfiler()
        >>> profiler.start()
        >>> 
        >>> # Your distributed training code
        >>> with profiler.profile_comm("all_reduce"):
        ...     dist.all_reduce(tensor)
        >>> 
        >>> profiler.stop()
        >>> profiler.print_summary()
    """
    
    def __init__(self):
        """Initialize communication profiler."""
        self.events: List[CommEvent] = []
        self.is_profiling = False
        
        # Distributed info
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
    
    def start(self):
        """Start profiling."""
        self.is_profiling = True
        self.events = []
    
    def stop(self):
        """Stop profiling."""
        self.is_profiling = False
    
    def profile_comm(self, comm_type: str, data_size_bytes: int = 0):
        """
        Context manager for profiling a communication operation.
        
        Args:
            comm_type: Type of communication (e.g., "all_reduce", "all_gather")
            data_size_bytes: Size of data being communicated
        
        Example:
            >>> with profiler.profile_comm("all_reduce", tensor.numel() * tensor.element_size()):
            ...     dist.all_reduce(tensor)
        """
        return CommContext(self, CommType(comm_type), data_size_bytes)
    
    def record_event(self, event: CommEvent):
        """Record a communication event."""
        if self.is_profiling:
            self.events.append(event)
    
    def get_statistics(self) -> Dict[CommType, CommProfile]:
        """
        Get communication statistics grouped by operation type.
        
        Returns:
            Dictionary mapping CommType to CommProfile
        """
        stats: Dict[CommType, List[CommEvent]] = {}
        
        # Group events by type
        for event in self.events:
            if event.comm_type not in stats:
                stats[event.comm_type] = []
            stats[event.comm_type].append(event)
        
        # Calculate statistics for each type
        profiles = {}
        for comm_type, events in stats.items():
            total_time = sum(e.duration for e in events)
            total_data = sum(e.data_size_bytes for e in events)
            num_calls = len(events)
            
            avg_time = total_time / num_calls if num_calls > 0 else 0.0
            avg_bandwidth = sum(e.bandwidth_gbps for e in events) / num_calls if num_calls > 0 else 0.0
            
            profiles[comm_type] = CommProfile(
                comm_type=comm_type,
                num_calls=num_calls,
                total_time=total_time,
                total_data_bytes=total_data,
                avg_time=avg_time,
                avg_bandwidth_gbps=avg_bandwidth,
            )
        
        return profiles
    
    def print_summary(self):
        """Print communication profiling summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("Communication Profiling Summary")
        print("="*70)
        print(f"Rank: {self.rank}/{self.world_size}")
        print(f"Total events: {len(self.events)}")
        print("")
        
        for comm_type, profile in sorted(stats.items(), key=lambda x: x[1].total_time, reverse=True):
            print(profile)
            print("")
        
        # Overall statistics
        total_comm_time = sum(p.total_time for p in stats.values())
        total_data = sum(p.total_data_bytes for p in stats.values())
        
        print(f"Total communication time: {total_comm_time:.3f}s")
        print(f"Total data transferred: {total_data / 1e9:.2f} GB")
        print("="*70 + "\n")


class CommContext:
    """Context manager for profiling a single communication operation."""
    
    def __init__(self, profiler: CommunicationProfiler, comm_type: CommType, data_size_bytes: int):
        self.profiler = profiler
        self.comm_type = comm_type
        self.data_size_bytes = data_size_bytes
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record event."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        if self.start_time is not None:
            event = CommEvent(
                comm_type=self.comm_type,
                start_time=self.start_time,
                end_time=end_time,
                data_size_bytes=self.data_size_bytes,
                rank=self.profiler.rank,
                group_size=self.profiler.world_size,
            )
            self.profiler.record_event(event)
        
        return False

