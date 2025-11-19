# NeMo Fusion - Quick Start Guide

Get started with NeMo Fusion in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/abdullah1kazi/nemo-fusion.git
cd nemo-fusion

# Install with UV (recommended)
uv sync

# Verify installation
uv run python verify_setup.py
```

## Basic Usage

### 1. Auto-Optimize Parallelism Strategy

```python
from nemo_fusion.parallelism import AutoParallelOptimizer

# Create optimizer
optimizer = AutoParallelOptimizer()

# Optimize for your model
strategy = optimizer.optimize(
    num_params=70_000_000_000,  # 70B parameters
    num_layers=80,
    hidden_size=8192,
    num_gpus=16,
    gpu_memory_gb=80,
    batch_size=1,
    sequence_length=4096,
    num_attention_heads=64,
)

# Get NeMo configuration
nemo_config = strategy.to_nemo_config()
print(f"Optimal strategy: TP={strategy.tensor_parallel}, PP={strategy.pipeline_parallel}")
```

### 2. Profile Your Training

```python
from nemo_fusion.profiling import GPUProfiler, DistributedProfiler

# Profile GPU utilization
gpu_profiler = GPUProfiler()
with gpu_profiler:
    # Your training loop
    pass

gpu_profiler.print_report()

# Detect bottlenecks
dist_profiler = DistributedProfiler()
dist_profiler.start()

# Training...

dist_profiler.stop()
report = dist_profiler.analyze()
print(f"Efficiency: {report.overall_efficiency:.2%}")
```

### 3. Mixed Precision Training

```python
from nemo_fusion.optimization import MixedPrecisionConfig

# For H100 GPUs with FP8
config = MixedPrecisionConfig.for_h100(use_fp8=True)

# For A100 GPUs with BF16
config = MixedPrecisionConfig.for_a100()

# Convert to NeMo format
nemo_config = config.to_nemo_config()
```

### 4. Multi-Modal Training

```python
from nemo_fusion.multimodal import MultiModalFusion

# Create fusion layer
fusion = MultiModalFusion(
    modality_dims={"text": 768, "image": 1024},
    output_dim=512,
    fusion_type="cross_attention",
    num_heads=8,
)

# Use in your model
features = {
    "text": text_embeddings,    # [batch, seq, 768]
    "image": image_embeddings,  # [batch, seq, 1024]
}
fused = fusion(features)  # [batch, seq, 512]
```

## Run Examples

```bash
# LLaMA 70B optimization
uv run python examples/llama_70b_optimization.py

# Profiling demo
uv run python examples/profiling_demo.py

# Multi-modal training
uv run python examples/multimodal_training.py
```

## Run Tests

```bash
# All tests
uv run pytest tests/

# Specific module
uv run pytest tests/unit/test_parallelism.py

# With coverage
uv run pytest tests/ --cov=nemo_fusion
```

## Run Benchmarks

```bash
# Compare parallelism strategies
uv run python benchmarks/parallelism_comparison.py

# View results
cat benchmarks/PERFORMANCE_RESULTS.md
```

## Common Use Cases

### Optimize for Your Hardware

```python
from nemo_fusion.parallelism import HybridParallelStrategy

# Small model (< 10B params) on 8 GPUs
strategy = HybridParallelStrategy.for_small_model(num_gpus=8)

# Medium model (10B-70B) on 8 GPUs
strategy = HybridParallelStrategy.for_medium_model(num_gpus=8)

# Large model (70B-175B) on 16 GPUs
strategy = HybridParallelStrategy.for_large_model(num_gpus=16)

# Very large model (> 175B) on 64 GPUs
strategy = HybridParallelStrategy.for_very_large_model(num_gpus=64)
```

### Gradient Accumulation

```python
from nemo_fusion.optimization import GradientOptimizer

grad_optimizer = GradientOptimizer(
    model=model,
    optimizer=optimizer,
    gradient_accumulation_steps=4,
    gradient_clip_val=1.0,
)

for batch in dataloader:
    loss = model(batch)
    grad_optimizer.backward(loss)
    
    if grad_optimizer.should_step():
        grad_optimizer.step()
        grad_optimizer.zero_grad()
```

### Checkpoint Management

```python
from nemo_fusion.optimization import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    save_interval=1000,
    keep_last_n=3,
)

if manager.should_save(step):
    manager.save_checkpoint(
        step=step,
        model=model,
        optimizer=optimizer,
    )
```

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation instructions
- **[Parallelism Guide](docs/PARALLELISM_GUIDE.md)** - Understanding parallelism strategies
- **[Optimization Tricks](docs/OPTIMIZATION_TRICKS.md)** - Performance optimization techniques
- **[Contributing](CONTRIBUTING.md)** - How to contribute

## Getting Help

- Check the [examples/](examples/) directory for complete working examples
- Read the [documentation](docs/) for detailed guides
- Open an issue on GitHub for bugs or questions

## Next Steps

1. ✅ Run `verify_setup.py` to ensure everything is installed
2. 📖 Read the [Parallelism Guide](docs/PARALLELISM_GUIDE.md)
3. 🚀 Try the [examples](examples/)
4. 🧪 Run the [tests](tests/)
5. 📊 Check the [benchmarks](benchmarks/)

Happy training! 🎉

