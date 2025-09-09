import queue
import torch
import time
from collections import deque
from block_manager import BlockManager, Block
from cache_engine import CacheEngine
from async_transfer_engine import AsyncTransferEngine
from typing import List, Set, Callable, Tuple


class AsyncPrefetcher(AsyncTransferEngine):
    def __init__(
        self, block_manager: BlockManager, cache_engine: CacheEngine, transfer_unit: int
    ):
        """
        Args:
            block_manager: 块管理器
            cache_engine: 缓存引擎
            transfer_unit: 传输单位
        """
        super().__init__(
            block_manager,
            cache_engine,
            transfer_unit,
            src_device=torch.device("cpu"),
            dst_device=torch.device("cuda"),
            name="AsyncPrefetcher",
        )

        # 特有的变量
        self.watermark = self.block_manager.watermark
        self.prefetch_queue: deque[int] = deque()
        self._prefetched_layers: Set[int] = set()

        self.start()

        # 当engine中的generate中计算完一个层后，需要把该层的层号发给prefetcher
        # prefetcher需要从该层的下一层开始，检查该层以后最近的层中的在CPU中的块
        # 把这些块按照单位unit进行流式拷贝到GPU上，直到该层所需要的最少快都在GPU上
        # 这时，该层开始计算，并终止预取进程预取该层的块

    def _get_prefetch_layers(self, current_layer: int) -> List[int]:
        """
        可根据系统负载动态调整预取层数。
        当前实现为静态：预取 current_layer + 1 和 +2。
        """
        # TODO 这里需要根据watermark来决定预取的层数
        # TODO 这里预取的层数是循环的，预取完最后一层后就应该预取第一层了
        return [
            layer
            for layer in range(current_layer + 1, current_layer + 3)
            if layer < self.block_manager.num_attn_layers
            and layer not in self._prefetched_layers
        ]

    def notify(self, layer: int):
        with self._condition:
            # TODO 这里需要根据watermark来决定预取的层数
            for future_layer in self._get_prefetch_layers(layer):
                self.prefetch_queue.append(future_layer)
                self._prefetched_layers.add(future_layer)
            self._condition.notify()

    def _should_wait(self) -> bool:
        return not self.prefetch_queue and not self._shutdown

    def _get_task(self):
        layer = self.prefetch_queue.popleft()
        self._prefetched_layers.remove(layer)
        return layer

    def _transfer(self, task):
        layer = task
        sorted_blocks = self.block_manager.predict_next_layer_needed_blocks(layer)
        num_blocks = len(sorted_blocks)
        current_step = 0

        if num_blocks == 0:
            return

        # ✳️ 如果 GPU 块数量触及水位线，则可以触发 offload（你可以自定义调用）
        if self.block_manager.gpu_free_block_num() < int(
            self.block_manager.num_gpu_blocks * self.watermark
        ):
            print(f"⚠️ Layer {layer} 预取前触及 GPU 水位线，建议触发卸载策略")

        while current_step < num_blocks:
            blocks = sorted_blocks[current_step : current_step + self.transfer_unit]
            if not blocks:
                break
            try:
                plan = self.block_manager.get_prefetch_plan(blocks)
                print(f"prefetch plan for layer {layer} at step {current_step}: {plan}")
            except RuntimeError as e:
                print(f"🟥 Layer {layer} prefetch failed: {e}")
                break

            self._prefetch_unit(plan, blocks)
            current_step += self.transfer_unit

        print(f"✅ Layer {layer} 预取完成")

    def _prefetch_unit(self, plan: List[tuple[int, int]], blocks: List[Block]):
        """生成预取计划的tensor,并执行异步拷贝，拷贝完成后更新block的device"""
        blocks_to_prefetch = torch.tensor(plan, device="cpu", dtype=torch.int64).view(
            -1, 2
        )

        def on_transfer_complete():
            self.block_manager.update_block_device_prefetch(plan, blocks)

        self.cache_engine.transfer_blocks_async(
            blocks_to_prefetch,
            self.src_device,
            self.dst_device,
            self.transfer_stream,  # type: ignore
            callback_fn=on_transfer_complete,
            add_event=self.event_monitor.add_event,
        )
