"""
Multi-Modal Fusion Layers for NeMo Fusion.

Provides cross-modal attention and fusion mechanisms for
combining different modalities (text, image, audio, video).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for fusing different modalities.
    
    Allows one modality to attend to another modality, enabling
    information flow between different input types.
    
    Example:
        >>> # Text attending to image features
        >>> cross_attn = CrossModalAttention(
        ...     query_dim=768,  # Text embedding dim
        ...     key_value_dim=1024,  # Image embedding dim
        ...     num_heads=12
        ... )
        >>> text_features = torch.randn(batch, seq_len, 768)
        >>> image_features = torch.randn(batch, num_patches, 1024)
        >>> fused_features = cross_attn(text_features, image_features)
    """
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            query_dim: Dimension of query modality
            key_value_dim: Dimension of key/value modality
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(key_value_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(key_value_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor [batch, query_len, query_dim]
            key_value: Key/value tensor [batch, kv_len, key_value_dim]
            attention_mask: Attention mask [batch, query_len, kv_len]
        
        Returns:
            Output tensor [batch, query_len, query_dim]
        """
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, self.query_dim)
        output = self.out_proj(attn_output)
        
        return output


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module combining multiple modalities.
    
    Supports various fusion strategies:
    - Early fusion: Concatenate features
    - Late fusion: Separate processing then combine
    - Cross-attention fusion: Bidirectional cross-modal attention
    
    Example:
        >>> fusion = MultiModalFusion(
        ...     modality_dims={"text": 768, "image": 1024, "audio": 512},
        ...     output_dim=768,
        ...     fusion_type="cross_attention"
        ... )
        >>> features = {
        ...     "text": torch.randn(batch, seq_len, 768),
        ...     "image": torch.randn(batch, num_patches, 1024),
        ...     "audio": torch.randn(batch, audio_len, 512),
        ... }
        >>> fused = fusion(features)
    """
    
    def __init__(
        self,
        modality_dims: dict,
        output_dim: int,
        fusion_type: str = "cross_attention",
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-modal fusion.
        
        Args:
            modality_dims: Dictionary mapping modality name to dimension
            output_dim: Output dimension
            fusion_type: "concat", "cross_attention", or "gated"
            num_heads: Number of attention heads (for cross_attention)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.modality_names = list(modality_dims.keys())
        
        if fusion_type == "concat":
            # Simple concatenation + projection
            total_dim = sum(modality_dims.values())
            self.fusion_proj = nn.Linear(total_dim, output_dim)
        
        elif fusion_type == "cross_attention":
            # Cross-attention between modalities
            self.cross_attentions = nn.ModuleDict()
            for mod_name in self.modality_names:
                self.cross_attentions[mod_name] = CrossModalAttention(
                    query_dim=output_dim,
                    key_value_dim=modality_dims[mod_name],
                    num_heads=num_heads,
                    dropout=dropout,
                )
            
            # Project each modality to output_dim
            self.modality_projs = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
        
        elif fusion_type == "gated":
            # Gated fusion with learnable gates
            self.modality_projs = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
            self.gate = nn.Linear(output_dim * len(modality_dims), len(modality_dims))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_features: dict) -> torch.Tensor:
        """
        Fuse multiple modalities.
        
        Args:
            modality_features: Dictionary mapping modality name to features
        
        Returns:
            Fused features [batch, seq_len, output_dim]
        """
        if self.fusion_type == "concat":
            # Concatenate all modalities
            features = [modality_features[name] for name in self.modality_names]
            concat_features = torch.cat(features, dim=-1)
            return self.fusion_proj(concat_features)
        
        elif self.fusion_type == "cross_attention":
            # Project all modalities to output_dim
            projected = {
                name: self.modality_projs[name](modality_features[name])
                for name in self.modality_names
            }
            
            # Average as initial representation
            fused = torch.stack(list(projected.values())).mean(dim=0)
            
            # Apply cross-attention from each modality
            for name in self.modality_names:
                fused = fused + self.cross_attentions[name](fused, modality_features[name])
            
            return self.dropout(fused)
        
        elif self.fusion_type == "gated":
            # Project all modalities
            projected = [
                self.modality_projs[name](modality_features[name])
                for name in self.modality_names
            ]
            
            # Compute gates
            concat_proj = torch.cat(projected, dim=-1)
            gates = F.softmax(self.gate(concat_proj), dim=-1)
            
            # Weighted sum
            fused = sum(
                gates[..., i:i+1] * projected[i]
                for i in range(len(projected))
            )
            
            return self.dropout(fused)

