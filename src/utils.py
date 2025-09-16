"""Utility functions for the application."""

import threading
from typing import Dict, Optional


class ThreadSafeDict:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[int, int] = {}

    def update(
        self,
        old_gid: Optional[int] = None,
        new_gid: Optional[int] = None,
        seq_id: Optional[int] = None,
    ) -> None:
        """
        一个接口处理三种情况：
        1. add: old_gid=None, new_gid!=None, seq_id!=None
        2. remove: old_gid!=None, new_gid=None
        3. migrate: old_gid!=None, new_gid!=None, seq_id=None
        """
        with self._lock:
            # ✅ 添加
            if old_gid is None and new_gid is not None and seq_id is not None:
                self._data[new_gid] = seq_id
                print(f"[GidToSeq] ADD gid={new_gid} -> seq={seq_id}")
                return

            # ✅ 删除
            if old_gid is not None and new_gid is None:
                if old_gid in self._data:
                    sid = self._data[old_gid]
                    del self._data[old_gid]
                    print(f"[GidToSeq] REMOVE gid={old_gid} (seq={sid})")
                else:
                    print(f"[Warning][GidToSeq] REMOVE failed, gid={old_gid} not found")
                return

            # ✅ 迁移
            if old_gid is not None and new_gid is not None and seq_id is None:
                if old_gid not in self._data:
                    print(
                        f"[Warning][GidToSeq] MIGRATE failed, old_gid={old_gid} not found new_gid={new_gid}"
                    )
                    return
                sid = self._data[old_gid]
                del self._data[old_gid]
                self._data[new_gid] = sid
                print(f"[GidToSeq] MIGRATE {old_gid} -> {new_gid} (seq={sid})")
                return

            raise ValueError(
                f"Invalid arguments: old_gid={old_gid}, new_gid={new_gid}, seq_id={seq_id}"
            )

    def get(self, gid: int):
        with self._lock:
            return self._data.get(gid, None)

    def items(self):
        with self._lock:
            return list(self._data.items())

    # 重载len()函数
    def __len__(self):
        with self._lock:
            return len(self._data)


def is_pin_memory_available() -> bool:
    """Check if pin memory is available."""
    return True
