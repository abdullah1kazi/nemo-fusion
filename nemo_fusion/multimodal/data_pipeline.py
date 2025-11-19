"""
Multi-Modal Data Pipeline for NeMo Fusion.

Provides data loading and preprocessing for multi-modal training
with support for text, image, audio, and video modalities.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    
    name: str
    data_type: str  # "text", "image", "audio", "video"
    max_length: Optional[int] = None
    preprocessing_fn: Optional[Callable] = None
    tokenizer: Optional[Any] = None


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset supporting multiple input modalities.
    
    Example:
        >>> dataset = MultiModalDataset(
        ...     data=[
        ...         {
        ...             "text": "A cat sitting on a mat",
        ...             "image": image_tensor,
        ...             "label": 1
        ...         },
        ...         # ... more samples
        ...     ],
        ...     modality_configs={
        ...         "text": ModalityConfig(name="text", data_type="text", max_length=512),
        ...         "image": ModalityConfig(name="image", data_type="image"),
        ...     }
        ... )
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        modality_configs: Dict[str, ModalityConfig],
        transform: Optional[Callable] = None,
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data: List of data samples (each is a dict with modality keys)
            modality_configs: Configuration for each modality
            transform: Optional transform to apply to samples
        """
        self.data = data
        self.modality_configs = modality_configs
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with processed modality features
        """
        sample = self.data[idx]
        processed = {}
        
        for modality_name, config in self.modality_configs.items():
            if modality_name not in sample:
                continue
            
            data = sample[modality_name]
            
            # Apply preprocessing
            if config.preprocessing_fn is not None:
                data = config.preprocessing_fn(data)
            
            # Apply tokenization for text
            if config.data_type == "text" and config.tokenizer is not None:
                data = config.tokenizer(
                    data,
                    max_length=config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                data = {k: v.squeeze(0) for k, v in data.items()}
            
            # Convert to tensor if not already
            if not isinstance(data, torch.Tensor):
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                elif isinstance(data, (list, tuple)):
                    data = torch.tensor(data)
            
            processed[modality_name] = data
        
        # Add labels if present
        if "label" in sample:
            processed["label"] = torch.tensor(sample["label"])
        
        # Apply transform
        if self.transform is not None:
            processed = self.transform(processed)
        
        return processed


class MultiModalDataLoader:
    """
    Data loader for multi-modal datasets with custom collation.
    
    Handles batching of different modalities with potentially
    different sequence lengths and data types.
    
    Example:
        >>> dataloader = MultiModalDataLoader(
        ...     dataset=dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     text_features = batch["text"]
        ...     image_features = batch["image"]
        ...     labels = batch["label"]
    """
    
    def __init__(
        self,
        dataset: MultiModalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize multi-modal data loader.
        
        Args:
            dataset: MultiModalDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for multi-modal batches.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Batched dictionary
        """
        if not batch:
            return {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        collated = {}
        for key in keys:
            # Stack all tensors for this key
            tensors = [sample[key] for sample in batch if key in sample]
            
            if not tensors:
                continue
            
            # Handle different tensor types
            if isinstance(tensors[0], dict):
                # Nested dictionary (e.g., tokenizer output)
                collated[key] = {
                    k: torch.stack([t[k] for t in tensors])
                    for k in tensors[0].keys()
                }
            else:
                # Regular tensor
                collated[key] = torch.stack(tensors)
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

