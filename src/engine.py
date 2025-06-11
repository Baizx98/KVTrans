"""This is engine with async kv streaming block transfer."""

from block_manager import BlockManager
from worker import Worker
from async_offloader import AsyncOffloader
from config import CacheConfig, ModelConfig, DeviceConfig


class Engine:
    """Engine class for managing the KV Cache with async streaming block transfer."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        """Initialize the Engine with cache, model, and device configurations."""
        self.cache_config = cache_config
        self.model_config = model_config
        self.device_config = device_config

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.dtype = model_config.dtype
        self.device = device_config.device

        self.block_manager = BlockManager(cache_config, model_config, device_config)
        self.worker = Worker(cache_config, model_config, device_config)
        self.block_transfer_unit = cache_config.transfer_unit
        self.async_offloader = AsyncOffloader(
            self.block_manager,
            self.worker.cache_engine,
            self.block_transfer_unit,
        )

    def generate(self):
        self.step()

    def step(self):
        for layer in range(self.model_config.num_attn_layers):
            self.layer_step(layer)

    def layer_step(self, layer: int):
        self.worker.execute_model(input_data=None)  # Replace with actual input data

        # 产生当前层计算完毕的事件

        # 通知上一层的卸载任务并在其完成当前传输的原子步骤后终止任务

        # 开始执行当前层的卸载任务,并自动终止上一层
        self.async_offloader.start_offload(layer)
