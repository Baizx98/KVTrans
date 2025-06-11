import threading
from block_manager import BlockManager
from cache_engine import CacheEngine
from typing import List, Optional
from utils import is_pin_memory_available


class AsyncOffloader:
    def __init__(
        self, block_manager: BlockManager, cache_engine: CacheEngine, transfer_unit: int
    ):
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit = transfer_unit

        self.offload_thread: Optional[threading.Thread] = None
        self.abort_event = threading.Event()
        self.lock = threading.Lock()

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
        # 如果想要动态步长，那就使用while循环
        # 我真傻，这里本来就适合while循环，为什么非要用for循环呢？！
        while current_step < num_blocks:
            if self.abort_event.is_set():
                print(f"🟡 Layer {layer} offload interrupted at step {current_step}.")
                break

            # 获取当前步长的块
            blocks = sorted_blocks[current_step : current_step + self.transfer_unit]
            if not blocks:
                break
            plan = self.block_manager.get_offload_plan(blocks)

            self._offload_unit(plan)
            current_step += self.transfer_unit

        plan = self.block_manager.get_offload_plan(layer)

        print(f"✅ Layer {layer} offload complete.")

    def _offload_unit(self, plan: List[tuple]):
        self.cache_engine.copy_blocks_async(plan)
