from block_manager import BlockManager, Block, BlockTable
from config import CacheConfig, ModelConfig, DeviceConfig
from typing import List, Dict, Optional
from engine import Engine
from worker import Worker
import time

SeqId = int
BlockId = int


if __name__ == "__main__":
    # cache_config = CacheConfig(
    #     block_size=32,
    #     num_gpu_blocks=64,
    #     num_cpu_blocks=64,
    #     watermark=0.5,
    #     transfer_unit=4,
    # )
    # model_config = ModelConfig(model_name="test_model", num_attn_layers=4)
    # device_config = DeviceConfig(device="cuda", device_id=0)
    # engine = Engine(
    #     cache_config=cache_config,
    #     model_config=model_config,
    #     device_config=device_config,
    # )
    pass
