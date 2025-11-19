"""Unit tests for optimization module."""

import pytest
import torch
import torch.nn as nn
from nemo_fusion.optimization import (
    MixedPrecisionConfig,
    PrecisionType,
    GradientOptimizer,
    GradientAccumulationScheduler,
    CheckpointManager,
    CheckpointCompressor,
)


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MixedPrecisionConfig()
        
        assert config.precision == PrecisionType.BF16
        assert config.master_weights_dtype == torch.float32
        assert config.use_loss_scaling is False  # BF16 doesn't need loss scaling
    
    def test_fp16_config(self):
        """Test FP16 configuration."""
        config = MixedPrecisionConfig(precision=PrecisionType.FP16)
        
        assert config.precision == PrecisionType.FP16
        assert config.use_loss_scaling is True  # FP16 requires loss scaling
        assert config.gradient_dtype == torch.float16
    
    def test_for_h100(self):
        """Test H100 optimized config."""
        config = MixedPrecisionConfig.for_h100(use_fp8=True)
        
        assert config.precision == PrecisionType.FP8
    
    def test_for_a100(self):
        """Test A100 optimized config."""
        config = MixedPrecisionConfig.for_a100()
        
        assert config.precision == PrecisionType.BF16
        assert config.use_loss_scaling is False
    
    def test_for_v100(self):
        """Test V100 optimized config."""
        config = MixedPrecisionConfig.for_v100()
        
        assert config.precision == PrecisionType.FP16
        assert config.use_loss_scaling is True
    
    def test_to_nemo_config(self):
        """Test conversion to NeMo config."""
        config = MixedPrecisionConfig(precision=PrecisionType.BF16)
        nemo_config = config.to_nemo_config()
        
        assert nemo_config["precision"] == "bf16"


class TestPrecisionType:
    """Tests for PrecisionType enum."""
    
    def test_bytes_per_param(self):
        """Test bytes per parameter calculation."""
        assert PrecisionType.FP32.bytes_per_param == 4
        assert PrecisionType.FP16.bytes_per_param == 2
        assert PrecisionType.BF16.bytes_per_param == 2
        assert PrecisionType.FP8.bytes_per_param == 1


class TestGradientOptimizer:
    """Tests for GradientOptimizer."""
    
    def test_initialization(self):
        """Test gradient optimizer initialization."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        grad_optimizer = GradientOptimizer(
            model=model,
            optimizer=optimizer,
            gradient_accumulation_steps=4,
        )
        
        assert grad_optimizer.gradient_accumulation_steps == 4
        assert grad_optimizer.current_step == 0
    
    def test_should_step(self):
        """Test should_step logic."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        grad_optimizer = GradientOptimizer(
            model=model,
            optimizer=optimizer,
            gradient_accumulation_steps=4,
        )
        
        # First 3 steps should not trigger optimizer step
        for i in range(3):
            grad_optimizer.current_step += 1
            assert not grad_optimizer.should_step()
        
        # 4th step should trigger
        grad_optimizer.current_step += 1
        assert grad_optimizer.should_step()


class TestGradientAccumulationScheduler:
    """Tests for GradientAccumulationScheduler."""
    
    def test_constant_schedule(self):
        """Test constant accumulation schedule."""
        scheduler = GradientAccumulationScheduler(initial_steps=4)

        for step in range(100):
            assert scheduler.get_accum_steps(step) == 4

    def test_custom_schedule(self):
        """Test custom schedule function."""
        def schedule_fn(step):
            return min(16, 4 * (step // 100 + 1))

        scheduler = GradientAccumulationScheduler(
            initial_steps=4,
            schedule_fn=schedule_fn,
        )

        assert scheduler.get_accum_steps(0) == 4
        assert scheduler.get_accum_steps(100) == 8
        assert scheduler.get_accum_steps(200) == 12

    def test_min_max_clamping(self):
        """Test min/max clamping."""
        def schedule_fn(step):
            return step  # Would grow unbounded

        scheduler = GradientAccumulationScheduler(
            schedule_fn=schedule_fn,
            min_steps=2,
            max_steps=8,
        )

        assert scheduler.get_accum_steps(1) == 2  # Clamped to min
        assert scheduler.get_accum_steps(100) == 8  # Clamped to max


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    def test_initialization(self, tmp_path):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_interval=100,
            keep_last_n=3,
        )
        
        assert manager.checkpoint_dir.exists()
        assert manager.save_interval == 100
        assert manager.keep_last_n == 3
    
    def test_should_save(self, tmp_path):
        """Test should_save logic."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_interval=100,
        )
        
        assert not manager.should_save(0)
        assert not manager.should_save(50)
        assert manager.should_save(100)
        assert manager.should_save(200)


class TestCheckpointCompressor:
    """Tests for CheckpointCompressor."""
    
    def test_compress_state_dict(self):
        """Test state dict compression."""
        state_dict = {
            "weight": torch.randn(10, 10, dtype=torch.float32),
            "bias": torch.randn(10, dtype=torch.float32),
        }
        
        compressed = CheckpointCompressor.compress_dict(state_dict)

        assert compressed["weight"].dtype == torch.float16
        assert compressed["bias"].dtype == torch.float16

    def test_decompress_state_dict(self):
        """Test state dict decompression."""
        state_dict = {
            "weight": torch.randn(10, 10, dtype=torch.float16),
            "bias": torch.randn(10, dtype=torch.float16),
        }

        decompressed = CheckpointCompressor.decompress_dict(state_dict)
        
        assert decompressed["weight"].dtype == torch.float32
        assert decompressed["bias"].dtype == torch.float32

