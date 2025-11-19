import time
from typing import List
from dataclasses import dataclass
import json
from nemo_fusion.parallelism import AutoParallelOptimizer, ModelConfig, HardwareConfig


@dataclass
class BenchmarkResult:
    model_name: str
    num_params: int
    num_gpus: int
    gpu_type: str
    strategy_name: str
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    context_parallel: int
    memory_per_gpu_gb: float
    expected_throughput: float
    efficiency_score: float
    optimization_time_ms: float


def benchmark_model(
    model_name: str,
    model_config: ModelConfig,
    hardware_configs: List[tuple],
    optimizer: AutoParallelOptimizer,
) -> List[BenchmarkResult]:
    results = []
    print(f"\nBenchmarking {model_name}...")

    for config_name, hw_config in hardware_configs:
        start_time = time.time()
        strategy = optimizer.optimize(
            num_params=model_config.num_params,
            num_layers=model_config.num_layers,
            hidden_size=model_config.hidden_size,
            num_gpus=hw_config.num_gpus,
            gpu_memory_gb=hw_config.gpu_memory_gb,
            batch_size=1,
            sequence_length=model_config.sequence_length,
            num_attention_heads=model_config.num_attention_heads,
            use_context_parallel=True,
        )
        optimization_time = (time.time() - start_time) * 1000

        result = BenchmarkResult(
            model_name=model_name,
            num_params=model_config.num_params,
            num_gpus=hw_config.num_gpus,
            gpu_type=hw_config.gpu_type,
            strategy_name=config_name,
            tensor_parallel=strategy.tensor_parallel,
            pipeline_parallel=strategy.pipeline_parallel,
            data_parallel=strategy.data_parallel,
            context_parallel=strategy.context_parallel,
            memory_per_gpu_gb=strategy.memory_per_gpu_gb,
            expected_throughput=strategy.expected_throughput,
            efficiency_score=strategy.efficiency_score,
            optimization_time_ms=optimization_time,
        )
        results.append(result)
        print(f"  {config_name:20s} | TP={strategy.tensor_parallel:2d} PP={strategy.pipeline_parallel:2d} "
              f"DP={strategy.data_parallel:2d} CP={strategy.context_parallel:2d} | "
              f"Mem={strategy.memory_per_gpu_gb:5.1f}GB | Eff={strategy.efficiency_score:5.1%}")
    return results


def main():
    print("\nNeMo Fusion Parallelism Strategy Comparison\n")

    models = {
        "GPT-3 (175B)": ModelConfig(175_000_000_000, 96, 12288, 96, 2048),
        "LLaMA 70B": ModelConfig(70_000_000_000, 80, 8192, 64, 4096),
        "LLaMA 13B": ModelConfig(13_000_000_000, 40, 5120, 40, 4096),
        "GPT-2 (1.5B)": ModelConfig(1_500_000_000, 48, 1600, 25, 1024),
    }

    hardware_configs = [
        ("16x H100", HardwareConfig(16, 80, "H100")),
        ("32x H100", HardwareConfig(32, 80, "H100")),
        ("64x H100", HardwareConfig(64, 80, "H100")),
    ]

    optimizer = AutoParallelOptimizer()
    all_results = []

    for model_name, model_config in models.items():
        results = benchmark_model(model_name, model_config, hardware_configs, optimizer)
        all_results.extend(results)

    print("\nSummary:")
    for model_name in models.keys():
        model_results = [r for r in all_results if r.model_name == model_name]
        print(f"  {model_name}: Best eff={max(r.efficiency_score for r in model_results):.1%}, "
              f"Best throughput={max(r.expected_throughput for r in model_results):.2f} tok/s")

    with open("benchmarks/results.json", "w") as f:
        json.dump([{
            "model": r.model_name,
            "gpus": r.num_gpus,
            "strategy": f"TP={r.tensor_parallel} PP={r.pipeline_parallel} DP={r.data_parallel} CP={r.context_parallel}",
            "mem_gb": r.memory_per_gpu_gb,
            "throughput": r.expected_throughput,
            "efficiency": r.efficiency_score,
        } for r in all_results], f, indent=2)

    print("\nResults saved to benchmarks/results.json")


if __name__ == "__main__":
    main()

