import torch
from nemo_fusion.parallelism import AutoParallelOptimizer, ModelConfig, HardwareConfig


def main():
    print("LLaMA 70B Parallelism Optimization\n")

    llama_70b_config = ModelConfig(
        num_params=70_000_000_000,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        sequence_length=4096,
        vocab_size=32000,
    )

    gpu_configs = [
        ("16x H100", HardwareConfig(num_gpus=16, gpu_memory_gb=80, gpu_type="H100")),
        ("32x H100", HardwareConfig(num_gpus=32, gpu_memory_gb=80, gpu_type="H100")),
        ("64x A100", HardwareConfig(num_gpus=64, gpu_memory_gb=80, gpu_type="A100")),
    ]

    optimizer = AutoParallelOptimizer()

    for config_name, hw_config in gpu_configs:
        print(f"\n{config_name}:")

        strategy = optimizer.optimize(
            num_params=llama_70b_config.num_params,
            num_layers=llama_70b_config.num_layers,
            hidden_size=llama_70b_config.hidden_size,
            num_gpus=hw_config.num_gpus,
            gpu_memory_gb=hw_config.gpu_memory_gb,
            batch_size=1,
            sequence_length=llama_70b_config.sequence_length,
            num_attention_heads=llama_70b_config.num_attention_heads,
            use_context_parallel=True,
        )

        print(f"  {strategy}")
        print(f"  NeMo config: {strategy.to_nemo_config()}")

        top_strategies = optimizer.get_top_strategies(3)
        if len(top_strategies) > 1:
            print(f"  Alternatives:")
        for i, alt in enumerate(top_strategies[1:4], 1):
            print(f"    {i}. {alt}")


if __name__ == "__main__":
    main()

