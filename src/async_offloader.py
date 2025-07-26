import queue
import threading
import torch
import time
from block_manager import BlockManager, Block
from cache_engine import CacheEngine
from async_transfer_engine import AsyncTransferEngine
from typing import List, Optional, Callable, Tuple


class AsyncOffloader(AsyncTransferEngine):
    def __init__(
        self, block_manager: BlockManager, cache_engine: CacheEngine, transfer_unit: int
    ):
        super().__init__(
            block_manager,
            cache_engine,
            transfer_unit,
            src_device=torch.device("cuda"),
            dst_device=torch.device("cpu"),
            name="AsyncOffloader",
        )

        # 特有的变量
        self._request_layer: Optional[int] = None
        self._abort_event = threading.Event()

        self.start()

    def notify(self, layer: int):
        # notify the offload thread to offload the layer
        self._abort_event.set()
        with self._condition:
            self._request_layer = layer
            self._condition.notify()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._condition.notify()
        self._monitor_shutdown = True

        self._abort_event.set()  # ✅ 中止当前 transfer 操作

        if self.monitor_thread:
            self.monitor_thread.join()
        if self.transfer_thread:
            self.transfer_thread.join()
        print(f"🔚 {self.name} shutdown complete.")

    def _should_wait(self) -> bool:
        # Custom condition to wait for offload requests
        return self._request_layer is None and not self._shutdown

    def _get_task(self):
        layer = self._request_layer
        self._request_layer = None
        self._abort_event.clear()
        return layer

    def _transfer(self, task):
        layer = task
        sorted_blocks = self.block_manager.get_layer_blocks_by_importance(layer)
        num_blocks = len(sorted_blocks)
        current_step = 0

        print(f"⬇️ Start offloading layer {layer} with {num_blocks} blocks...")

        while current_step < num_blocks:
            if self._abort_event.is_set() or self._shutdown:
                print(f"🟡 Layer {layer} offload interrupted at step {current_step}.")
                return

            blocks = sorted_blocks[current_step : current_step + self.transfer_unit]
            if not blocks:
                break

            plan = self.block_manager.get_offload_plan(blocks)
            print(f"🧠 Offload plan for layer {layer} at step {current_step}: {plan}")
            self._offload_unit(plan, blocks)
            current_step += self.transfer_unit

        print(f"✅ Offload complete for layer {layer}.")

    def _offload_unit(self, plan: List[tuple[int, int]], blocks: List[Block]):
        blocks_to_offload = torch.tensor(plan, device="cpu", dtype=torch.int64).view(
            -1, 2
        )

        # 闭包，简化函数参数传递
        def on_transfer_complete():
            self.block_manager.update_block_device_offload(plan, blocks)

        # self.cache_engine.offload_copy_blocks_async(
        #     blocks_to_offload,
        #     blocks,
        #     callback_fn=on_transfer_complete,
        # )
        self.cache_engine.transfer_blocks_async(
            blocks_to_offload,
            self.src_device,
            self.dst_device,
            self.transfer_stream,  # type: ignore
            callback_fn=on_transfer_complete,
        )

    def _event_monitor_worker(self):
        """这个线程专门负责检查CUDA事件是否完成，并在完成后执行回调（Offloader 专用）"""
        print("💡 Offload event monitor thread started.")
        pending_events: List[Tuple[torch.cuda.Event, Callable]] = []
        BATCH_SIZE = 16
        WAIT_TIME = 0.001

        while not self._monitor_shutdown:
            try:
                # 从队列中取任务，凑满一个 batch
                for _ in range(BATCH_SIZE - len(pending_events)):
                    event, callback_fn = (
                        self.cache_engine.offload_monitor_queue.get_nowait()
                    )
                    pending_events.append((event, callback_fn))
            except queue.Empty:
                pass

            if not pending_events:
                time.sleep(WAIT_TIME)
                continue

            # 检查哪些事件已经完成
            ready_indices = []
            for i, (event, _) in enumerate(pending_events):
                if event.query():
                    ready_indices.append(i)

            # 执行回调并移除完成的事件
            for idx in reversed(ready_indices):  # 倒序 pop 防止索引错乱
                _, callback_fn = pending_events.pop(idx)
                callback_fn()

            time.sleep(WAIT_TIME)
