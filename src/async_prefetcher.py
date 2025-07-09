import threading
import torch
import time
from block_manager import BlockManager, Block
from cache_engine import CacheEngine
from typing import List, Optional
from utils import is_pin_memory_available

class AsyncPrefetcher:
    def __init__(self, block_manager, cache_engine, transfer_unit):
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit = transfer_unit

        self.prefetch_thread:Optional[threading.Thread] = None
        self.abort_event = threading.Event()
        self.lock = threading.Lock()

        # 当engine中的generate中计算完一个层后，需要把该层的层号发给prefetcher
        # prefetcher需要从该层的下一层开始，检查该层以后最近的层中的在CPU中的块  
        # 把这些块按照单位unit进行流式拷贝到GPU上，直到该层所需要的最少快都在GPU上
        # 这时，该层开始计算，并终止预取进程预取该层的块

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

    def start_prefetch(self,layer:int):
        """在预取layer层的kv时,在第layer层进行计算之前,必须把该层所需要的最少KV块拷贝到GPU上"""
        # 其实也是
        with self.lock:
            if self.prefetch_thread and self.prefetch_thread.is_alive():
                self.abort_event.set()
                self.prefetch_thread.join()
                print("🟥 Previous prefetch task aborted.")
            self.abort_event.clear()
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(layer,))
            self.prefetch_thread.start()

    def _prefetch_worker(self,layer:int):
        sorted_blocks = self.block_manager.predict_next_layer_needed_blocks(layer)
        num_blocks = len(sorted_blocks)
        current_step = 0

        while current_step < num_blocks:
            if self.abort_event.is_set():
                print(f"🟡 Layer {layer} prefetch interrupted at step {current_step}.")
                break
            # self.transfer_unit = self.update_transfer_unit(num_blocks, current_step)
            print(f"tranfer unit is {self.transfer_unit}")
            # FIXME 此处切片索引是否会越界呢
            blocks = sorted_blocks[current_step : current_step + self.transfer_unit]
            if not blocks:
                break
            plan = self.block_manager.get_prefetch_plan(blocks)
            print(f"prefetch plan for layer {layer} at step {current_step}: {plan}")
            self._prefetch_unit(plan, blocks)
            current_step += self.transfer_unit
        print(f"✅ Layer {layer} prefetch complete.")
    
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
        print("💡 Prefetch event monitor thread started.")
        while True:
            # 检查是否有任务需要中止
            if self.abort_event.is_set() and self.prefetch_thread and not self.prefetch_thread.is_alive():
                # FIXME prefetch_thread 在切换时也会终止，这里有可能会意外触发改逻辑
                # 如果主 prefetch 线程已经中止且不活跃，可以考虑停止监控线程
                # 或者让它继续等待新的 prefetch 任务
                # 为简单起见，这里让它一直运行
                pass
            
            events_to_process = []
            with self.cache_engine.prefetch_callbacks_lock:
                # 遍历所有待处理的事件
                for event, callback_fn in list(
                    self.cache_engine.prefetch_completion_callbacks.items()
                ):
                    if event.query():  # 检查事件是否完成
                        events_to_process.append((event, callback_fn))
            
            for event, callback_fn in events_to_process:
                # 移除已完成的事件 这里是什么意思呢？NOTE
                with self.cache_engine.prefetch_callbacks_lock:
                    del self.cache_engine.prefetch_completion_callbacks[event]
                # 执行回调函数
                callback_fn()

            time.sleep(0.001)  # 短暂休眠，避免忙等待消耗过多 CPU