"""This is engine with async kv streaming block transfer."""

from block_manager import BlockManager
from scheduler import cleanup_batch, schedule_batch
from worker import Worker
from async_offloader import AsyncOffloader
from async_prefetcher import AsyncPrefetcher
from config import CacheConfig, ModelConfig, DeviceConfig
from sequence import Sequence
from typing import List


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
        self.async_prefetcher = AsyncPrefetcher(
            self.block_manager,
            self.worker.cache_engine,
            self.block_transfer_unit,
        )

        self.prefill_flag = True

    def generate(self, batch: List[Sequence]):
        while batch:
            # scheduled_batch 和 batch 是同一个对象
            scheduled_batch = schedule_batch(batch)
            print(f"scheduled batch nums: {len(scheduled_batch)}")
            # 推理
            self.step(batch)
            # 模拟新token生成和生成结束
            for sequence in batch:
                sequence.generate_new_token()
            # 清理 这里的两个batch不是同一个对象
            batch = cleanup_batch(batch)
        print("All sequences processed and cleaned up.")

    def step(self, batch: List[Sequence]):
        if self.prefill_flag:
            self.prefill_flag = False
            for sequence in batch:
                need_block_nums = sequence.seq_len + 1  # +1 for the initial block
                self.block_manager.allocate_gpu_blocks_for_all_layers(
                    sequence.seq_id, need_block_nums
                )
                sequence.block_num = need_block_nums
        # 此处分配prefill，一次性为所有层分配所有prompt的块+1，只有第一次step
        for layer in range(self.model_config.num_attn_layers):
            # 此处分配decode阶段，每一层的新块，第二次step开始
            if not self.prefill_flag:
                for sequence in batch:
                    self.block_manager.allocate_gpu_blocks_for_layer(
                        sequence.seq_id, 1, layer
                    )
            print(f"layer {layer}")
            print(f"token nums of batch: {batch[0].seq_len}")
            print(f"len of gid_to_seq: {len(self.block_manager.gid_to_seq)}")
            self.layer_step(batch, layer)
            print(f"cpu block num:{self.block_manager.cpu_free_block_num()}")
            print(f"gpu block num:{self.block_manager.gpu_free_block_num()}")

    def layer_step(self, batch: List[Sequence], layer: int):
        # 要保证该层的计算开始前，所需要的块已经到位
        # 需要有一个队列将CPU中的块按照使用它们的顺序排好队，
        # 从batch的当前层开始看，把
        # 判断当前层是否预取完毕,以及当前层是否正在预取，如果完毕，则开始预取将来最近一层需要的块，如果正在预取，则等待预取完成
        # 如果该层的kv cache not ready，说明预取线程一定在预取该层的kv cache
        self.block_manager.wait_for_kv_cache_ready(batch, layer)
        print(f"🔵 Starting layer {layer} step with {len(batch)} sequences.")
        self.worker.execute_model(input_data=batch)  # Replace with actual input data

        # 产生当前层计算完毕的事件

        # 通知上一层的卸载任务并在其完成当前传输的原子步骤后终止任务

        # 开始执行当前层的卸载任务,并自动终止上一层
        self.async_offloader.start_offload(layer)
        self.async_prefetcher.notify(layer)
        # 预取应该放在这里，它应该是一个常驻的线程，单纯地通过事件来同步
        # 也就是说，预取线程会一直运行，不断地将加下来所需要的数据从CPU传输到GPU
        # 在每层的计算开始前，阻塞当前层的预取任务，
        # 如何保证 有新的块供分配呢？答案是要预取的比卸载的要少
        # 还需要一个指标来说明当前超额分配了多少块，下一步或未来几步需要多少块，预取回来的块除了缓冲区以外
        # 需要把这些块也空出来，并且作为预取块数的指标
