import threading
import torch
import time
from block_manager import BlockManager, Block
from cache_engine import CacheEngine
from typing import List, Optional
from utils import is_pin_memory_available


class AsyncOffloader:
    def __init__(
        self, block_manager: BlockManager, cache_engine: CacheEngine, transfer_unit: int
    ):
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit: int = transfer_unit

        self._condition = threading.Condition()
        self._request_layer : Optional[int] = None
        self._shutdown = False
        self._monitor_shutdown = False
        self._abort_event = threading.Event()

        self.offload_thread: threading.Thread = threading.Thread(target = self._run, name="offload_thread")
        self.offload_thread.start()
        self.event_monitor_thread = threading.Thread(
            target=self._event_monitor_worker,
            name="offload_event_monitor_thread",
        )
        self.event_monitor_thread.start()

    def request_offload(self, layer: int):
        # notify the offload thread to offload the layer
        self._abort_event.set()
        with self._condition:
            self._request_layer = layer
            self._condition.notify()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._monitor_shutdown = True
            self._abort_event.set()
            self._condition.notify()
        self.offload_thread.join()
        self.event_monitor_thread.join()

    def _run(self):
        # offload worker loop
        while True:
            with self._condition:
                while self._request_layer is None and not self._shutdown:
                    self._condition.wait()

                if self._shutdown:
                    print("🔚 Offloader shutting down.")
                    break

                layer = self._request_layer
                self._request_layer = None
                self._abort_event.clear()
            if layer is not None:
                self._offload_layer(layer)
            else:
                # 抛出异常
                raise RuntimeError("No layer requested for offloading")


    def _offload_layer(self, layer: int):
        sorted_blocks = self.block_manager.get_layer_blocks_by_importance(layer)
        num_blocks = len(sorted_blocks)
        current_step = 0

        print(f"⬇️ Start offloading layer {layer} with {num_blocks} blocks...")

        while current_step < num_blocks:
            if self._abort_event.is_set():
                print(f"🟡 Layer {layer} offload interrupted at step {current_step}.")
                return

            blocks = sorted_blocks[current_step: current_step + self.transfer_unit]
            if not blocks:
                break

            plan = self.block_manager.get_offload_plan(blocks)
            print(f"🧠 Offload plan for layer {layer} at step {current_step}: {plan}")
            self._offload_unit(plan, blocks)
            current_step += self.transfer_unit

        print(f"✅ Offload complete for layer {layer}.")


    def update_transfer_unit(self, num_blocks: int, current_step: int) -> int:
        # 如果每次传输的处理开销太大，则增加传输单位，否则减小传输单位
        if is_pin_memory_available():
            self.transfer_unit = min(
                self.block_manager.num_gpu_blocks, self.transfer_unit * 2
            )
        else:
            self.transfer_unit = max(1, self.transfer_unit // 2)
        self.transfer_unit = min(self.transfer_unit, num_blocks - current_step)
        return self.transfer_unit

    def _offload_unit(self, plan: List[tuple[int, int]], blocks: List[Block]):
        blocks_to_offload = torch.tensor(plan, device="cpu", dtype=torch.int64).view(
            -1, 2
        )

        # 闭包，简化函数参数传递
        def on_transfer_complete():
            self.block_manager.update_block_device_offload(plan, blocks)

        self.cache_engine.offload_copy_blocks_async(
            blocks_to_offload,
            blocks,
            callback_fn=on_transfer_complete,
        )

    def _event_monitor_worker(self):
        """
        这个线程专门负责检查 CUDA 事件是否完成，并在完成后执行回调。
        """
        print("💡 Offload Event monitor thread started.")
        while not self._monitor_shutdown:
            events_to_process = []
            with self.cache_engine.offload_callbacks_lock:
                # 遍历所有待处理的事件
                for event, callback_fn in list(
                    self.cache_engine.offload_completion_callbacks.items()
                ):
                    if event.query():  # 检查事件是否完成
                        events_to_process.append((event, callback_fn))

            for event, callback_fn in events_to_process:
                # 移除已完成的事件 这里是什么意思呢？NOTE
                with self.cache_engine.offload_callbacks_lock:
                    del self.cache_engine.offload_completion_callbacks[event]
                # 执行回调函数
                callback_fn()

            time.sleep(0.001)  # 短暂休眠，避免忙等待消耗过多 CPU
