from block_manager import BlockManager, Block, BlockTable
from config import CacheConfig, ModelConfig, DeviceConfig
from typing import List, Dict, Optional
from engine import Engine
from worker import Worker
import time
import os 

from sequence import generate_random_batch

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


SeqId = int
BlockId = int


if __name__ == "__main__":
    # 初始化配置
    cache_config = CacheConfig(
        block_size=2,
        num_gpu_blocks=10000,
        num_cpu_blocks=50000,
        watermark=0.8,
        transfer_unit=4,
    )
    model_config = ModelConfig(
        model_name="example_model", num_attn_layers=12, num_kv_heads=2, head_size=4
    )
    device_config = DeviceConfig(device="cuda", device_id=0)
    # 初始化引擎
    engine = Engine(cache_config, model_config, device_config)
    # 随机生成一个请求批次
    batch = generate_random_batch(batch_size=4, seq_length=8)

    # 模仿generate的逻辑
    # 推理
    output = engine.generate(batch)
