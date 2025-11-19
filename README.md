# NeMo Fusion 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Advanced Parallelism and Optimization Toolkit for NVIDIA NeMo Framework**

NeMo Fusion is a high-performance extension toolkit for [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo) that provides advanced distributed training optimizations, intelligent parallelism strategies, and comprehensive profiling tools.

**Author**: [Abdullah Kazi](https://github.com/abdullah1kazi)

---

## 🎯 Features

### 🔧 Core Capabilities

- **Auto-Parallelism Optimizer**: Automatically determine optimal TP/PP/DP/CP configurations based on model size and hardware
- **Distributed Training Profiler**: Identify bottlenecks in GPU utilization, communication overhead, and memory usage
- **Memory-Efficient Attention**: Flash Attention and Ring Attention implementations compatible with NeMo
- **Mixed Precision Training**: FP8, FP16, BF16 training recipes optimized for H100/H200 GPUs
- **Multi-Modal Training Extensions**: Unified interface for text, image, and video modalities

### 📊 Performance Optimizations

- Hybrid parallelism strategies (TP + PP + DP + CP combinations)
- Communication overlap optimization
- Gradient accumulation optimizer
- Dynamic batch size scheduling
- Checkpoint compression utilities

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (for training)
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

#### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone the repository

```bash
git clone https://github.com/abdullah1kazi/nemo-fusion.git
cd nemo-fusion
```

#### 3. Install NeMo Fusion

**For users:**
```bash
uv sync
```

**For developers:**
```bash
uv sync --all-extras
```

**Quick install (without UV):**
```bash
pip install -e .
```

---

## 📖 Usage Examples

### Example 1: Auto-Optimize LLaMA 70B Training

```python
from nemo_fusion.parallelism import AutoParallelOptimizer
from nemo_fusion.optimization import OptimizedTrainer

# Define model and hardware configuration
optimizer = AutoParallelOptimizer()

strategy = optimizer.optimize(
    num_params=70e9,        # 70B parameters
    num_layers=80,
    hidden_size=8192,
    num_gpus=8,
    gpu_memory_gb=80,
    batch_size=32,
    sequence_length=2048
)

print(f"Optimal config: TP={strategy.tensor_parallel}, PP={strategy.pipeline_parallel}")
print(f"Expected memory: {strategy.memory_per_gpu_gb:.1f} GB/GPU")
print(f"Expected throughput: {strategy.expected_throughput:.0f} tokens/sec")
```

### Example 2: Profile Training Bottlenecks

```python
from nemo_fusion.profiling import DistributedProfiler

profiler = DistributedProfiler()

with profiler.profile():
    # Your training loop
    for batch in dataloader:
        output = model(batch)
        loss.backward()
        optimizer.step()

# Get detailed analysis
report = profiler.analyze()
print(report.bottlenecks)
print(report.recommendations)
```

### Example 3: Multi-Modal Training

```python
from nemo_fusion.multimodal import UnifiedMultiModalTrainer

trainer = UnifiedMultiModalTrainer(
    text_encoder="llama-7b",
    vision_encoder="vit-large",
    fusion_strategy="cross_attention"
)

trainer.fit(dataloader)
```

---

## 🛠️ Development

### Using UV for Development

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Run tests
uv run pytest tests/

# Run tests with coverage
make test-cov

# Format code
make format

# Run linters
make lint

# Build documentation
make docs
```

### Project Structure

```
nemo-fusion/
├── nemo_fusion/
│   ├── __init__.py
│   ├── parallelism/          # Auto-parallelism and hybrid strategies
│   │   ├── auto_parallel.py
│   │   ├── hybrid_parallel.py
│   │   └── memory_efficient.py
│   ├── profiling/            # Distributed training profilers
│   │   ├── bottleneck_analyzer.py
│   │   ├── gpu_profiler.py
│   │   └── comm_profiler.py
│   ├── optimization/         # Training optimizations
│   │   ├── mixed_precision.py
│   │   ├── gradient_optimization.py
│   │   └── checkpoint_utils.py
│   └── multimodal/          # Multi-modal extensions
│       ├── fusion_layers.py
│       ├── data_pipeline.py
│       └── unified_trainer.py
├── examples/                 # Usage examples
├── benchmarks/              # Performance benchmarks
├── tests/                   # Unit and integration tests
├── pyproject.toml          # UV/Python project config
├── Makefile                # Development commands
├── QUICKSTART.md           # Quick start guide
└── README.md
```

---

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Examples](examples/) - Working code examples
- [Tests](tests/) - Unit and integration tests

---

## 🎯 Alignment with NVIDIA NeMo

NeMo Fusion is designed to seamlessly integrate with NVIDIA NeMo Framework:

- ✅ Compatible with NeMo 2.0+ API
- ✅ Works with Megatron-Core parallelism strategies
- ✅ Supports NeMo's Tensor, Pipeline, Data, and Context Parallelism
- ✅ Integrates with PyTorch Lightning training loops
- ✅ Compatible with NeMo's checkpoint format

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `uv run pytest tests/` to verify
5. Submit a pull request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)

---

## 📧 Contact

**Author**: Abdullah Kazi
**GitHub**: [https://github.com/abdullah1kazi](https://github.com/abdullah1kazi)
**Project**: [https://github.com/abdullah1kazi/nemo-fusion](https://github.com/abdullah1kazi/nemo-fusion)

For questions and support, please open an issue on the [project repository](https://github.com/abdullah1kazi/nemo-fusion/issues).