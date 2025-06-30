"""CacheEngine class for managing the KV Cache."""

import threading
import torch
from config import CacheConfig, ModelConfig, DeviceConfig
from block_manager import Block
from utils import is_pin_memory_available
from typing import List, Tuple, Dict, Callable


class CacheEngine:
    """Manage the physical KV Cache."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        """Initialize the CacheEngine with cache, model, and device configurations."""
        self.cache_config = cache_config
        self.model_config = model_config
        self.device_config = device_config

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.dtype: torch.dtype = model_config.dtype
        self.device: str = device_config.device

        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device
        )
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        print(
            f"Allocated KV Cache: GPU blocks={self.num_gpu_blocks}, CPU blocks={self.num_cpu_blocks}, "
            f"GPU cache shape={self.gpu_cache.shape}, CPU cache shape={self.cpu_cache.shape}"
        )

        self.data_stream = torch.cuda.Stream()
        self.completion_callbacks: Dict[torch.cuda.Event, Callable] = {}
        self.callbacks_lock = threading.Lock()

    def _allocate_kv_cache(self, num_blocks, device) -> torch.Tensor:
        """Allocate KV cache on the specified device."""
        kv_cache_shape = self.cache_config.get_kv_cache_shape(
            num_blocks,
            self.model_config.num_kv_heads,
            self.model_config.head_size,
        )
        pin_memory: bool = is_pin_memory_available() if device == "cpu" else False
        kv_cache = torch.zeros(
            kv_cache_shape, dtype=self.dtype, pin_memory=pin_memory, device=device
        )
        return kv_cache

    def copy_blocks_async(
        self,
        blocks_to_offload: torch.Tensor,
        original_blocks: List[Block],
        callback_fn=None,
    ):
        src_block_ids = blocks_to_offload[:, 0]
        dst_block_ids = blocks_to_offload[:, 1]
        print(f"[AsyncCopy] Copy {len(src_block_ids)} blocks GPU→CPU.")

        with torch.cuda.stream(stream=self.data_stream):  # type: ignore
            tmp_tensor = self.gpu_cache[:, src_block_ids, :].contiguous()
            self.cpu_cache[:, dst_block_ids, :].copy_(tmp_tensor, non_blocking=True)
            event: torch.cuda.Event = torch.cuda.Event(blocking=False)  # type: ignore
            event.record(self.data_stream)
            if callback_fn:
                with self.callbacks_lock:
                    self.completion_callbacks[event] = callback_fn
