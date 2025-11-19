"""
Multi-modal module for NeMo Fusion.

Provides multi-modal training extensions:
- Cross-modal fusion layers
- Efficient multi-modal data pipelines
- Unified training interface
"""

from nemo_fusion.multimodal.fusion_layers import (
    CrossModalAttention,
    MultiModalFusion,
)
from nemo_fusion.multimodal.data_pipeline import (
    MultiModalDataLoader,
    MultiModalDataset,
    ModalityConfig,
)
from nemo_fusion.multimodal.unified_trainer import (
    UnifiedMultiModalTrainer,
    TrainingConfig,
)

__all__ = [
    "CrossModalAttention",
    "MultiModalFusion",
    "MultiModalDataLoader",
    "MultiModalDataset",
    "ModalityConfig",
    "UnifiedMultiModalTrainer",
    "TrainingConfig",
]

