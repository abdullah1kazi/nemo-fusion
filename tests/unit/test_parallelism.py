"""Unit tests for parallelism module."""

import pytest
from nemo_fusion.parallelism import (
    AutoParallelOptimizer,
    ModelConfig,
    HardwareConfig,
    ParallelStrategy,
    HybridParallelStrategy,
    ParallelismType,
)


class TestAutoParallelOptimizer:
    """Tests for AutoParallelOptimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        optimizer = AutoParallelOptimizer()
        assert optimizer is not None
        assert optimizer.strategies == []
    
    def test_optimize_small_model(self):
        """Test optimization for small model prefers data parallelism."""
        optimizer = AutoParallelOptimizer()
        
        strategy = optimizer.optimize(
            num_params=1_000_000_000,  # 1B
            num_layers=24,
            hidden_size=1024,
            num_gpus=8,
            gpu_memory_gb=80,
            batch_size=1,
            sequence_length=2048,
            num_attention_heads=16,
        )
        
        assert strategy is not None
        assert strategy.data_parallel >= 4  # Should prefer DP
        assert strategy.tensor_parallel <= 2
    
    def test_optimize_large_model(self):
        """Test optimization for large model uses hybrid parallelism."""
        optimizer = AutoParallelOptimizer()
        
        strategy = optimizer.optimize(
            num_params=70_000_000_000,  # 70B
            num_layers=80,
            hidden_size=8192,
            num_gpus=16,
            gpu_memory_gb=80,
            batch_size=1,
            sequence_length=4096,
            num_attention_heads=64,
        )
        
        assert strategy is not None
        assert strategy.tensor_parallel > 1  # Should use TP
        assert strategy.memory_per_gpu_gb > 0
        assert strategy.memory_per_gpu_gb <= 80
    
    def test_strategy_to_nemo_config(self):
        """Test conversion to NeMo config format."""
        strategy = ParallelStrategy(
            tensor_parallel=4,
            pipeline_parallel=2,
            data_parallel=2,
            context_parallel=1,
        )
        
        config = strategy.to_nemo_config()
        
        assert config["tensor_model_parallel_size"] == 4
        assert config["pipeline_model_parallel_size"] == 2
        assert config["data_parallel_size"] == 2
        assert config["context_parallel_size"] == 1
    
    def test_get_top_strategies(self):
        """Test getting top N strategies."""
        optimizer = AutoParallelOptimizer()
        
        optimizer.optimize(
            num_params=13_000_000_000,
            num_layers=40,
            hidden_size=5120,
            num_gpus=8,
            gpu_memory_gb=80,
            batch_size=1,
            sequence_length=2048,
            num_attention_heads=40,
        )
        
        top_strategies = optimizer.get_top_strategies(n=3)
        assert len(top_strategies) <= 3
        
        # Strategies should be sorted by efficiency
        if len(top_strategies) > 1:
            assert top_strategies[0].efficiency_score >= top_strategies[1].efficiency_score


class TestHybridParallelStrategy:
    """Tests for HybridParallelStrategy."""
    
    def test_for_small_model(self):
        """Test strategy for small model."""
        strategy = HybridParallelStrategy.for_small_model(num_gpus=8)
        
        assert strategy.tensor_parallel == 1
        assert strategy.pipeline_parallel == 1
        assert strategy.data_parallel == 8
    
    def test_for_medium_model(self):
        """Test strategy for medium model."""
        strategy = HybridParallelStrategy.for_medium_model(num_gpus=8)
        
        assert strategy.tensor_parallel > 1
        assert strategy.sequence_parallel is True
    
    def test_for_large_model(self):
        """Test strategy for large model."""
        strategy = HybridParallelStrategy.for_large_model(num_gpus=16)
        
        assert strategy.tensor_parallel > 1
        assert strategy.pipeline_parallel > 1
        assert strategy.virtual_pipeline_parallel is not None
    
    def test_total_gpus(self):
        """Test total GPU calculation."""
        strategy = HybridParallelStrategy(
            name="Test",
            tensor_parallel=4,
            pipeline_parallel=2,
            data_parallel=2,
        )
        
        assert strategy.total_gpus == 16
    
    def test_to_nemo_config(self):
        """Test conversion to NeMo config."""
        strategy = HybridParallelStrategy(
            name="Test",
            tensor_parallel=4,
            pipeline_parallel=2,
            data_parallel=2,
            sequence_parallel=True,
        )
        
        config = strategy.to_nemo_config()
        
        assert config["tensor_model_parallel_size"] == 4
        assert config["pipeline_model_parallel_size"] == 2
        assert config["data_parallel_size"] == 2
        assert config["sequence_parallel"] is True


class TestParallelismType:
    """Tests for ParallelismType enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert ParallelismType.TENSOR.value == "tensor"
        assert ParallelismType.PIPELINE.value == "pipeline"
        assert ParallelismType.DATA.value == "data"
        assert ParallelismType.CONTEXT.value == "context"
        assert ParallelismType.EXPERT.value == "expert"
        assert ParallelismType.SEQUENCE.value == "sequence"

