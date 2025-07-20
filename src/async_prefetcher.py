import queue
import threading
import torch
import time
from collections import deque
from queue import Queue
from block_manager import BlockManager, Block
from cache_engine import CacheEngine
from typing import List, Optional, Set, Callable, Tuple
from utils import is_pin_memory_available


class AsyncPrefetcher:
    def __init__(
        self, block_manager: BlockManager, cache_engine: CacheEngine, transfer_unit: int
    ):
        """
        Args:
            block_manager: 块管理器
            cache_engine: 缓存引擎
            transfer_unit: 传输单位
        """
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit = transfer_unit
        self.watermark = self.block_manager.watermark

        self.prefetch_queue: deque[int] = deque()
        self._condition = threading.Condition()
        self._shutdown = False
        self._monitor_shutdown = False
        self._prefetched_layers: Set[int] = set()

        self.prefetch_thread: Optional[threading.Thread] = threading.Thread(
            target=self._run, daemon=True, name="prefetch_thread"
        )
        self.prefetch_thread.start()

        self.event_monitor_thread: Optional[threading.Thread] = threading.Thread(
            target=self._event_monitor_worker,
            daemon=True,
            name="prefetch_event_monitor_thread",
        )
        self.event_monitor_thread.start()

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
            l
            for l in range(current_layer + 1, current_layer + 3)
            if l < self.block_manager.num_attn_layers
            and l not in self._prefetched_layers
        ]

    def notify(self, layer: int):
        with self._condition:
            # TODO 这里需要根据watermark来决定预取的层数
            for future_layer in self._get_prefetch_layers(layer):
                self.prefetch_queue.append(future_layer)
                self._prefetched_layers.add(future_layer)
            self._condition.notify()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._condition.notify()
        self._monitor_shutdown = True
        if self.event_monitor_thread:
            self.event_monitor_thread.join()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        print("🟢 AsyncPrefetcher shutdown complete.")

    def update_transfer_unit(self, num_blocks: int, current_step: int) -> int:
        """根据传输带宽利用率，更新传输单位，如果PCIe带宽利用率低，则增加传输单位，否则减小传输单位"""
        # FIXME: 需要根据传输带宽利用率来更新传输单位
        if is_pin_memory_available():
            self.transfer_unit = min(
                self.block_manager.num_gpu_blocks, self.transfer_unit * 2
            )
        else:
            self.transfer_unit = max(1, self.transfer_unit // 2)
        self.transfer_unit = min(self.transfer_unit, num_blocks - current_step)
        return self.transfer_unit

    def _run(self):
        while True:
            with self._condition:
                while not self.prefetch_queue and not self._shutdown:
                    self._condition.wait()
                if self._shutdown:
                    break
                layer = self.prefetch_queue.popleft()
                self._prefetched_layers.remove(layer)
            print(f"🟢 Layer {layer} prefetch started.")
            sorted_blocks = self.block_manager.predict_next_layer_needed_blocks(layer)
            num_blocks = len(sorted_blocks)
            current_step = 0

            if num_blocks == 0:
                continue

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
                    print(
                        f"prefetch plan for layer {layer} at step {current_step}: {plan}"
                    )
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

        self.cache_engine.prefetch_copy_blocks_async(
            blocks_to_prefetch,
            blocks,
            callback_fn=on_transfer_complete,
        )

    def _event_monitor_worker(self):
        """这个线程专门负责检查CUDA事件是否完成，并在完成后执行回调"""
        # TODO 添加定时回调处理或批量回调处理机制，减少调度和轮询开销
        print("💡 Prefetch event monitor thread started.")
        pending_events: List[Tuple[torch.cuda.Event, Callable]] = []
        BATCH_SIZE = 16
        WAIT_TIME = 0.001
        while not self._monitor_shutdown:
            # print("🟢 prefetch callback running")
            # TODO 检查是否有任务需要中止
            try:
                for _ in range(BATCH_SIZE - len(pending_events)):
                    event, callback_fn = (
                        self.cache_engine.prefetch_monitor_queue.get_nowait()
                    )
                    pending_events.append((event, callback_fn))
            except queue.Empty:
                pass

            if not pending_events:
                time.sleep(WAIT_TIME)
                continue

            for event, callback_fn in pending_events:
                while not event.query():
                    time.sleep(WAIT_TIME)

            ready_indices = []
            for i, (event, _) in enumerate(pending_events):
                if event.query():
                    ready_indices.append(i)

            for idx in reversed(ready_indices):
                _, callback_fn = pending_events.pop(idx)
                callback_fn()

            time.sleep(WAIT_TIME)
