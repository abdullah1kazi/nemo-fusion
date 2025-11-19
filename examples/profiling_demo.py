import torch
import torch.nn as nn
import time
from nemo_fusion.profiling import GPUProfiler, DistributedProfiler, CommunicationProfiler


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)


def demo_gpu_profiling():
    print("\nGPU Profiling Demo")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU profiling demo.")
        return

    device = torch.device("cuda")
    model = SimpleModel(2048).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    profiler = GPUProfiler(sample_interval=0.1)

    profiler.start()
    for step in range(50):
        batch = torch.randn(32, 512, 2048, device=device)
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        profiler.sample()
    profiler.stop()
    profiler.print_report()


def demo_bottleneck_analysis():
    print("\nBottleneck Analysis Demo")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping bottleneck analysis demo.")
        return

    device = torch.device("cuda")
    model = SimpleModel(2048).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    profiler = DistributedProfiler(enable_detailed_profiling=True)

    with profiler.profile():
        for step in range(30):
            batch = torch.randn(32, 512, 2048, device=device)
            loss = model(batch).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 5 == 0:
                profiler.record_communication_time(0.01)
            profiler.record_gpu_utilization(0.65 + (step % 10) * 0.02)
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
                profiler.record_memory_usage(mem)

    print(profiler.analyze())
    for i, rec in enumerate(profiler.get_recommendations(), 1):
        print(f"{i}. {rec}")


def demo_communication_profiling():
    print("\nCommunication Profiling Demo")

    profiler = CommunicationProfiler()
    profiler.start()

    print("\nSimulating distributed communication operations...")

    for i in range(10):
        with profiler.profile_comm("all_reduce", data_size_bytes=1024*1024*100):
            time.sleep(0.01)
        if i % 3 == 0:
            with profiler.profile_comm("all_gather", data_size_bytes=1024*1024*50):
                time.sleep(0.005)
        if i % 5 == 0:
            with profiler.profile_comm("broadcast", data_size_bytes=1024*1024*10):
                time.sleep(0.002)

    profiler.stop()
    profiler.print_summary()


def main():
    print("\nNeMo Fusion Profiling Demonstrations\n")
    demo_gpu_profiling()
    demo_bottleneck_analysis()
    demo_communication_profiling()


if __name__ == "__main__":
    main()

