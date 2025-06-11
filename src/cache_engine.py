"""CacheEngine class for managing the KV Cache."""

import threading
import torch
from config import CacheConfig, ModelConfig, DeviceConfig
from utils import is_pin_memory_available
from typing import List, Tuple


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

    def copy_blocks_async(self, plan: List[Tuple[str, int, str, int]]):
        threads = []

        def copy_block(src_dev, src_pid, dst_dev, dst_pid):
            src_tensor = self.gpu_cache if src_dev == "gpu" else self.cpu_cache
            dst_tensor = self.gpu_cache if dst_dev == "gpu" else self.cpu_cache
            dst_tensor[:, dst_pid] = (
                src_tensor[:, src_pid].detach().to(dst_tensor.device)
            )

        for src_dev, src_pid, dst_dev, dst_pid in plan:
            t = threading.Thread(
                target=copy_block, args=(src_dev, src_pid, dst_dev, dst_pid)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
