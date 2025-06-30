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

        self.offload_thread: Optional[threading.Thread] = None
        self.abort_event = threading.Event()
        self.lock = threading.Lock()
        self.event_monitor_thread = threading.Thread(
            target=self._event_monitor_worker, daemon=True
        )
        self.event_monitor_thread.start()

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

    def start_offload(self, layer: int):
        with self.lock:
            # 中止已有任务
            if self.offload_thread and self.offload_thread.is_alive():
                self.abort_event.set()
                self.offload_thread.join()

            self.abort_event.clear()
            self.offload_thread = threading.Thread(
                target=self._offload_worker, args=(layer,)
            )
            self.offload_thread.start()

    def _offload_worker(self, layer: int):
        sorted_blocks = self.block_manager.get_layer_blocks_by_importance(layer)

        num_blocks = len(sorted_blocks)
        current_step = 0

        while current_step < num_blocks:
            if self.abort_event.is_set():
                print(f"🟡 Layer {layer} offload interrupted at step {current_step}.")
                break

            # 获取当前步长的块
            # self.transfer_unit = self.update_transfer_unit(num_blocks, current_step)
            print(f"tranfer unit is {self.transfer_unit}")
            blocks = sorted_blocks[current_step : current_step + self.transfer_unit]
            if not blocks:
                break
            # 这里返回GPU块和CPU块的物理id
            plan = self.block_manager.get_offload_plan(blocks)
            print(f"offload plan for layer {layer} at step {current_step}: {plan}")
            self._offload_unit(plan, blocks)
            current_step += self.transfer_unit
        print(f"✅ Layer {layer} offload complete.")

    def _offload_unit(self, plan: List[tuple[int, int]], blocks: List[Block]):
        blocks_to_offload = torch.tensor(plan, device="cpu", dtype=torch.int64).view(
            -1, 2
        )

        # 闭包，简化函数参数传递
        def on_transfer_complete():
            self.block_manager.update_block_device(plan, blocks)

        self.cache_engine.copy_blocks_async(
            blocks_to_offload,
            blocks,
            callback_fn=on_transfer_complete,
        )

    def _event_monitor_worker(self):
        """
        这个线程专门负责检查 CUDA 事件是否完成，并在完成后执行回调。
        """
        print("💡 Event monitor thread started.")
        while True:
            # 检查是否有任务需要中止
            if self.abort_event.is_set() and not self.offload_thread.is_alive():
                # 如果主 offload 线程已经中止且不活跃，可以考虑停止监控线程
                # 或者让它继续等待新的 offload 任务
                # 为简单起见，这里让它一直运行
                pass

            events_to_process = []
            with self.cache_engine.callbacks_lock:
                # 遍历所有待处理的事件
                for event, callback_fn in list(
                    self.cache_engine.completion_callbacks.items()
                ):
                    if event.query():  # 检查事件是否完成
                        events_to_process.append((event, callback_fn))

            for event, callback_fn in events_to_process:
                # 移除已完成的事件 这里是什么意思呢？NOTE
                with self.cache_engine.callbacks_lock:
                    del self.cache_engine.completion_callbacks[event]
                # 执行回调函数
                callback_fn()

            time.sleep(0.001)  # 短暂休眠，避免忙等待消耗过多 CPU
