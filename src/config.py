"""This module contains the configuration for the application."""

from typing import Tuple
import torch


class CacheConfig:
    """Configuration for the KV Cache."""

    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 8,
        num_cpu_blocks: int = 2,
        watermark: float = 0.5,
        transfer_unit: int = 2,
    ) -> None:
        """Initialize the CacheConfig with default values."""
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.watermark = watermark

        self.transfer_unit: int = (
            transfer_unit  # Number of blocks to transfer in one go
        )

    def get_kv_cache_shape(
        self, num_blocks: int, num_kv_heads: int, head_size: int
    ) -> Tuple[int, ...]:
        """Get the shape of the KV cache."""
        return (2, num_blocks, self.block_size * num_kv_heads * head_size)


class ModelConfig:
    """Configuration for the model."""

    def __init__(
        self,
        model_name="default_model",
        num_attn_layers: int = 12,
        num_kv_heads: int = 8,
        head_size: int = 64,
    ) -> None:
        """Initialize the ModelConfig with default values."""
        self.model_name = model_name
        self.num_attn_layers = num_attn_layers
        self.dtype: torch.dtype = torch.float16  # Default dtype for model weights
        self.num_kv_heads: int = num_kv_heads  # Number of key-value heads
        self.head_size: int = head_size


class DeviceConfig:
    """Configuration for the device."""

    def __init__(self, device: str = "cpu", device_id=0) -> None:
        """Initialize the DeviceConfig with default values."""
        self.device = device
        self.device_id = device_id
