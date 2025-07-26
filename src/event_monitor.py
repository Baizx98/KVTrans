import threading
import queue
import time
import torch
from typing import Callable, Tuple, List


class EventMonitor:
    _instance = None
    _lock = threading.Lock()
    _ref_count = 0

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            cls._ref_count += 1
            return cls._instance

    def _init(self):
        self._monitor_shutdown = False
        self._queues: List[queue.Queue] = []
        self._pending_events: List[Tuple[torch.cuda.Event, Callable]] = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("✅ EventMonitor started.")

    def register_queue(self, q: queue.Queue):
        self._queues.append(q)

    def unregister(self):
        with self._lock:
            self._ref_count -= 1
            if self._ref_count == 0:
                self.shutdown()

    def shutdown(self):
        self._monitor_shutdown = True
        self._thread.join()
        EventMonitor._instance = None
        EventMonitor._ref_count = 0
        print("❎ EventMonitor shutdown.")

    def _monitor_loop(self):
        BATCH_SIZE = 16
        WAIT_TIME = 0.001

        while not self._monitor_shutdown:
            for q in self._queues:
                try:
                    for _ in range(BATCH_SIZE):
                        event, callback_fn = q.get_nowait()
                        self._pending_events.append((event, callback_fn))
                except queue.Empty:
                    continue

            # 过滤已完成事件
            ready_indices = [
                i for i, (event, _) in enumerate(self._pending_events) if event.query()
            ]

            for idx in reversed(ready_indices):
                _, callback = self._pending_events.pop(idx)
                callback()

            time.sleep(WAIT_TIME)
