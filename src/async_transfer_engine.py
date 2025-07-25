import threading
import time
import torch
from typing import Callable, List, Optional, Tuple
from block_manager import Block, BlockManager
from cache_engine import CacheEngine
from utils import is_pin_memory_available


class AsyncTransferEngine:
    def __init__(
        self,
        block_manager: BlockManager,
        cache_engine: CacheEngine,
        transfer_unit: int,
        src_device: torch.device,
        dst_device: torch.device,
        name: str,
    ) -> None:
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit = transfer_unit
        self.name = name

        self.src_device = src_device
        self.dst_device = dst_device

        self._condition = threading.Condition()
        self._shutdown = False
        self._monitor_shutdown = False

        self.transfer_thread = threading.Thread(target=self._run, name=f"{name}_thread")
        self.monitor_thread = threading.Thread(
            target=self._event_monitor_worker, name=f"{name}_monitor_thread"
        )
        self.transfer_thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._monitor_shutdown = True
            self._condition.notify()
        self.transfer_thread.join()
        self.monitor_thread.join()
        print(f"🔚 {self.name} shutdown complete.")

    def update_transfer_unit(self, num_blocks: int, current_step: int) -> int:
        if is_pin_memory_available():
            self.transfer_unit = min(
                self.block_manager.num_gpu_blocks, self.transfer_unit * 2
            )
        else:
            self.transfer_unit = max(1, self.transfer_unit // 2)
        self.transfer_unit = min(self.transfer_unit, num_blocks - current_step)
        return self.transfer_unit

    # 自定义的判断条件
    def _should_wait(self) -> bool:
        return not self._shutdown

    def _get_task(self):
        raise NotImplementedError

    def _transfer(self, task):
        raise NotImplementedError

    def _run(self):
        while True:
            with self._condition:
                while self._should_wait():
                    self._condition.wait()
                if self._shutdown:
                    break
                task = self._get_task()

            if task:
                self._transfer(task)

    def _event_monitor_worker(self):
        raise NotImplementedError

    def _transfer_unit(self, plan: List[Block], blocks: List[Block]):
        """
        Transfer a unit of blocks according to the offload plan.
        This method should be implemented in subclasses.
        """
        blocks_to_transfer = torch.tensor(plan, device="cpu", dtype=torch.int64).view(
            -1, 2
        )

        def on_transfer_complete():
            self.block_manager.update_blocks_after_transfer()

        self.cache_engine.transfer_blocks_async(
            blocks_to_transfer,
            self.src_device,
            self.dst_device,
            callback_fn=on_transfer_complete,
        )
