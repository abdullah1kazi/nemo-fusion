"""Unit tests for multimodal module."""

import pytest
import torch
import torch.nn as nn
from nemo_fusion.multimodal import (
    CrossModalAttention,
    MultiModalFusion,
    MultiModalDataset,
    MultiModalDataLoader,
    ModalityConfig,
)


class TestCrossModalAttention:
    """Tests for CrossModalAttention."""
    
    def test_initialization(self):
        """Test cross-modal attention initialization."""
        attn = CrossModalAttention(
            query_dim=768,
            key_value_dim=1024,
            num_heads=8,
        )
        
        assert attn.query_dim == 768
        assert attn.key_value_dim == 1024
        assert attn.num_heads == 8
        assert attn.head_dim == 768 // 8
    
    def test_forward(self):
        """Test forward pass."""
        attn = CrossModalAttention(
            query_dim=768,
            key_value_dim=1024,
            num_heads=8,
        )
        
        batch_size = 2
        query_len = 10
        kv_len = 20
        
        query = torch.randn(batch_size, query_len, 768)
        key_value = torch.randn(batch_size, kv_len, 1024)
        
        output = attn(query, key_value)
        
        assert output.shape == (batch_size, query_len, 768)
    
    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        attn = CrossModalAttention(
            query_dim=768,
            key_value_dim=1024,
            num_heads=8,
        )
        
        batch_size = 2
        query_len = 10
        kv_len = 20
        
        query = torch.randn(batch_size, query_len, 768)
        key_value = torch.randn(batch_size, kv_len, 1024)
        # Attention mask should be [batch, num_heads, query_len, kv_len] or broadcastable
        attention_mask = torch.zeros(batch_size, 1, query_len, kv_len)
        
        output = attn(query, key_value, attention_mask)
        
        assert output.shape == (batch_size, query_len, 768)


class TestMultiModalFusion:
    """Tests for MultiModalFusion."""
    
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = MultiModalFusion(
            modality_dims={"text": 768, "image": 1024},
            output_dim=512,
            fusion_type="concat",
        )
        
        batch_size = 2
        seq_len = 10
        
        features = {
            "text": torch.randn(batch_size, seq_len, 768),
            "image": torch.randn(batch_size, seq_len, 1024),
        }
        
        output = fusion(features)
        
        assert output.shape == (batch_size, seq_len, 512)
    
    def test_cross_attention_fusion(self):
        """Test cross-attention fusion."""
        fusion = MultiModalFusion(
            modality_dims={"text": 768, "image": 1024},
            output_dim=768,
            fusion_type="cross_attention",
            num_heads=8,
        )
        
        batch_size = 2
        seq_len = 10
        
        features = {
            "text": torch.randn(batch_size, seq_len, 768),
            "image": torch.randn(batch_size, seq_len, 1024),
        }
        
        output = fusion(features)
        
        assert output.shape == (batch_size, seq_len, 768)
    
    def test_gated_fusion(self):
        """Test gated fusion."""
        fusion = MultiModalFusion(
            modality_dims={"text": 768, "image": 1024},
            output_dim=768,
            fusion_type="gated",
        )
        
        batch_size = 2
        seq_len = 10
        
        features = {
            "text": torch.randn(batch_size, seq_len, 768),
            "image": torch.randn(batch_size, seq_len, 1024),
        }
        
        output = fusion(features)
        
        assert output.shape == (batch_size, seq_len, 768)


class TestMultiModalDataset:
    """Tests for MultiModalDataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        data = [
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 0},
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 1},
        ]
        
        modality_configs = {
            "text": ModalityConfig(name="text", data_type="text"),
            "image": ModalityConfig(name="image", data_type="image"),
        }
        
        dataset = MultiModalDataset(
            data=data,
            modality_configs=modality_configs,
        )
        
        assert len(dataset) == 2
    
    def test_getitem(self):
        """Test getting item from dataset."""
        data = [
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 0},
        ]
        
        modality_configs = {
            "text": ModalityConfig(name="text", data_type="text"),
            "image": ModalityConfig(name="image", data_type="image"),
        }
        
        dataset = MultiModalDataset(
            data=data,
            modality_configs=modality_configs,
        )
        
        sample = dataset[0]
        
        assert "text" in sample
        assert "image" in sample
        assert "label" in sample
        assert sample["text"].shape == (10, 768)
        assert sample["image"].shape == (20, 1024)


class TestMultiModalDataLoader:
    """Tests for MultiModalDataLoader."""
    
    def test_initialization(self):
        """Test dataloader initialization."""
        data = [
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 0},
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 1},
        ]
        
        modality_configs = {
            "text": ModalityConfig(name="text", data_type="text"),
            "image": ModalityConfig(name="image", data_type="image"),
        }
        
        dataset = MultiModalDataset(
            data=data,
            modality_configs=modality_configs,
        )
        
        dataloader = MultiModalDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        )
        
        assert len(dataloader) == 1  # 2 samples / batch_size 2
    
    def test_iteration(self):
        """Test iterating through dataloader."""
        data = [
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 0},
            {"text": torch.randn(10, 768), "image": torch.randn(20, 1024), "label": 1},
        ]
        
        modality_configs = {
            "text": ModalityConfig(name="text", data_type="text"),
            "image": ModalityConfig(name="image", data_type="image"),
        }
        
        dataset = MultiModalDataset(
            data=data,
            modality_configs=modality_configs,
        )
        
        dataloader = MultiModalDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        )
        
        for batch in dataloader:
            assert "text" in batch
            assert "image" in batch
            assert "label" in batch
            assert batch["text"].shape == (2, 10, 768)
            assert batch["image"].shape == (2, 20, 1024)
            assert batch["label"].shape == (2,)

