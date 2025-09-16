""""""

from collections import deque

from typing import List, Dict, Optional, Deque, FrozenSet, Tuple
from config import CacheConfig, ModelConfig, DeviceConfig
from sequence import Sequence
from utils import ThreadSafeDict
import torch
import time
import threading

BlockId = int
SeqId = int


# Block's state
class BlockState:
    # 一共有ready和transferring两种状态
    READY = 1
    TRANSFERRING = 0


class Block:
    def __init__(self, block_id: BlockId):
        self.block_id = block_id
        self.state = BlockState.READY  # TODO 初始状态为 READY

    def ready(self):
        """将块状态设置为 READY"""
        self.state = BlockState.READY

    def transferring(self):
        """将块状态设置为 TRANSFERRING"""
        self.state = BlockState.TRANSFERRING


class BlockTable:
    def __init__(self, block_size: int, blocks: Optional[List[Block]] = None):
        self.block_size = block_size
        if blocks is None:
            blocks = []
        self.blocks = blocks


class BlockManager:
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        self.watermark = cache_config.watermark
        self.num_attn_layers = model_config.num_attn_layers
        self._buffer_blocks = 0

        self.watermark_blocks = int(self.watermark * self.num_gpu_blocks)

        block_ids = list(range(self.num_gpu_blocks + self.num_cpu_blocks))
        gpu_block_ids = block_ids[: self.num_gpu_blocks]
        cpu_block_ids = block_ids[self.num_gpu_blocks :]

        self._cpu_free_block_indices: Deque[BlockId] = deque(cpu_block_ids)
        self._gpu_free_block_indices: Deque[BlockId] = deque(gpu_block_ids)
        self._cpu_all_block_indices: FrozenSet[BlockId] = frozenset(cpu_block_ids)
        self._gpu_all_block_indices: FrozenSet[BlockId] = frozenset(gpu_block_ids)

        self.layer_block_tables: List[Dict[SeqId, BlockTable]] = [
            {} for _ in range(self.num_attn_layers)
        ]
        self.gid_to_seq: ThreadSafeDict = ThreadSafeDict()
        # 记录每个序列在卸载和预取后是否已经ready可以计算
        self.seq_layer_is_ready: List[Dict[SeqId, bool]] = [
            {} for _ in range(self.num_attn_layers)
        ]

        self._lock = threading.Lock()

    @property
    def buffer_blocks(self) -> int:
        return self._buffer_blocks

    @buffer_blocks.setter
    def buffer_blocks(self, value: int) -> None:
        if value < 0:
            raise ValueError("Buffer blocks cannot be negative")
        self._buffer_blocks = value

    def can_allocate_blocks(self, device: torch.device, n: int) -> bool:
        if device.type == "cuda":
            return self._can_allocate_gpu_blocks(n)
        elif device.type == "cpu":
            return self._can_allocate_cpu_blocks(n)
        raise ValueError(f"Invalid device type: {device.type}")

    def allocate_block(self, device: torch.device) -> Block:
        with self._lock:
            if device.type == "cuda":
                return self._allocate_gpu_block()
            elif device.type == "cpu":
                return self._allocate_cpu_block()
        raise ValueError(f"Invalid device type: {device.type}")

    def allocate_block_id(self, device: torch.device) -> BlockId:
        with self._lock:
            if device.type == "cuda":
                return self._allocate_gpu_block_id()
            elif device.type == "cpu":
                return self._allocate_cpu_block_id()
        raise ValueError(f"Invalid device type: {device.type}")

    def free_block(self, block: Block) -> None:
        with self._lock:
            if self.is_gpu_block(block.block_id):
                self._free_gpu_block(block)
            elif self.is_cpu_block(block.block_id):
                self._free_cpu_block(block)
            else:
                raise ValueError(f"Invalid block id: {block.block_id}")

    def free_block_id(self, block_id: BlockId) -> None:
        with self._lock:
            if self.is_gpu_block(block_id):
                self._free_gpu_block_id(block_id)
            elif self.is_cpu_block(block_id):
                self._free_cpu_block_id(block_id)
            else:
                raise ValueError(f"Invalid block id: {block_id}")

    def _can_allocate_gpu_blocks(self, n: int) -> bool:
        # TODO watermark and buffer
        return len(self._gpu_free_block_indices) >= n

    def _can_allocate_cpu_blocks(self, n: int) -> bool:
        # TODO watermark and buffer
        return len(self._cpu_free_block_indices) >= n

    def _allocate_gpu_block_id(self) -> BlockId:
        return (
            self._gpu_free_block_indices.popleft()
            if self._gpu_free_block_indices
            else -1
        )

    def _allocate_cpu_block_id(self) -> BlockId:
        return (
            self._cpu_free_block_indices.popleft()
            if self._cpu_free_block_indices
            else -1
        )

    def _allocate_gpu_block(self) -> Block:
        block_id = self._allocate_gpu_block_id()
        if block_id == -1:
            raise RuntimeError("No available GPU blocks")
        # Block Pool
        return Block(block_id)

    def _allocate_cpu_block(self) -> Block:
        block_id = self._allocate_cpu_block_id()
        if block_id == -1:
            raise RuntimeError("No available CPU blocks")
        return Block(block_id)

    def _free_gpu_block(self, block: Block):
        gid = block.block_id
        self._gpu_free_block_indices.append(gid)
        del block

    def _free_gpu_block_id(self, block_id: BlockId):
        self._gpu_free_block_indices.append(block_id)

    def _free_cpu_block(self, block: Block):
        gid = block.block_id
        self._cpu_free_block_indices.append(gid)
        del block

    def _free_cpu_block_id(self, block_id: BlockId):
        self._cpu_free_block_indices.append(block_id)

    def get_device_and_pid(self, gid: BlockId) -> Tuple[torch.device, int]:
        if gid >= 0 and gid < self.num_gpu_blocks:
            return torch.device("cuda"), gid
        elif (
            gid >= self.num_gpu_blocks
            and gid < self.num_gpu_blocks + self.num_cpu_blocks
        ):
            return torch.device("cpu"), gid - self.num_gpu_blocks
        raise ValueError(f"Invalid gid: {gid}")

    def get_gid(self, device: torch.device, pid: int) -> BlockId:
        if device.type == "cuda":
            return pid
        elif device.type == "cpu":
            return self.num_gpu_blocks + pid
        raise ValueError(f"Invalid device: {device}")

    def get_layer_blocks_by_importance(self, layer: int) -> List[Block]:
        with self._lock:
            # Dummy: importance = block_id
            # 应该从GPU块中挑选
            # TODO 对每个请求分别处理，应该接收
            # TODO 这里应该对请求也加一层筛选，来兼容perfix cache
            if layer >= self.num_attn_layers:
                return []
            all_blocks: List[Block] = []
            for table in self.layer_block_tables[layer].values():
                all_blocks.extend(table.blocks)
            # FIXME 这里只判断是GPU Block还不行，要排除掉TRANSFERRING状态的块
            all_blocks = [
                block
                for block in all_blocks
                if self.is_gpu_block(block.block_id)
                and block.state != BlockState.TRANSFERRING
            ]
            return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def predict_next_layer_needed_blocks(self, layer: int) -> List[Block]:
        with self._lock:
            # 应该从CPU块中挑选
            if layer >= self.num_attn_layers:
                return []
            all_blocks: List[Block] = []
            for table in self.layer_block_tables[layer].values():
                all_blocks.extend(table.blocks)
            all_blocks = [
                block
                for block in all_blocks
                if self.is_cpu_block(block.block_id)
                and block.state != BlockState.TRANSFERRING
            ]
            return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def get_transfer_plan(
        self, blocks: List[Block], src_device: torch.device, dst_device: torch.device
    ) -> List[Tuple[int, int]]:
        """
        1. 将源块标记为 TRANSFERRING 状态
        2. 分配目标块并将其标记为 TRANSFERRING 状态
        3. 返回块的物理 ID 映射
        """
        # with self._lock:  # 保证多线程安全
        physical_block_id_mapping: List[Tuple[int, int]] = []
        block_num = len(blocks)
        # 检查目标设备是否有足够的空闲块
        if not self.can_allocate_blocks(dst_device, block_num):
            raise RuntimeError(
                f"Not enough blocks available for transfer from {src_device} to {dst_device}"
            )
        for block in blocks:
            src_block_id = block.block_id
            _, src_pid = self.get_device_and_pid(src_block_id)
            block.transferring()  # 源块标记为传输中
            # 分配目标块
            dst_block_id = self.allocate_block_id(dst_device)
            _, dst_pid = self.get_device_and_pid(dst_block_id)
            # 记录物理 ID 映射
            physical_block_id_mapping.append((src_pid, dst_pid))
        return physical_block_id_mapping

    def update_blocks_after_transfer(
        self,
        plan: List[Tuple[int, int]],
        original_blocks: List[Block],
        src_device: torch.device,
        dst_device: torch.device,
    ) -> None:
        """
        根据传输计划更新块的设备信息和映射。

        :param plan: [(src_pid, dst_pid), ...]
        :param original_blocks: 原始 Block 列表（会被更新 block_id）
        :param src_device: 源设备 (torch.device("cuda") 或 torch.device("cpu"))
        :param dst_device: 目标设备
        """
        # 这里加了大锁，但实际上可以使用单独的加锁的函数来处理
        with self._lock:
            for i, (src_pid, dst_pid) in enumerate(plan):
                src_gid = self.get_gid(src_device, src_pid)
                dst_gid = self.get_gid(dst_device, dst_pid)

                # 更新 block 的全局 id
                original_blocks[i].block_id = dst_gid

                self.gid_to_seq.update(old_gid=src_gid, new_gid=dst_gid)

                # 释放源设备的块 ID
                if src_device.type == "cuda":
                    self._free_gpu_block_id(src_gid)
                elif src_device.type == "cpu":
                    self._free_cpu_block_id(src_gid)
                else:
                    raise ValueError(f"Invalid src_device {src_device}")
                original_blocks[i].ready()  # 传输完成，标记为 READY

    def kv_cache_ready(self, batch: List[Sequence], layer: int) -> bool:
        # TODO 这里要阻塞的呀 这里判断是否所有序列的kv cache都ready的逻辑有问题
        for seq in batch:
            if seq.seq_id not in self.layer_block_tables[layer]:
                return False
        return True

    def wait_for_kv_cache_ready(self, batch: List[Sequence], layer: int) -> None:
        while not self.kv_cache_ready(batch, layer):
            print("kv not ready阻塞……")
            time.sleep(0.001)

    def cpu_free_block_num(self) -> int:
        return len(self._cpu_free_block_indices)

    def gpu_free_block_num(self) -> int:
        return len(self._gpu_free_block_indices)

    def is_cpu_block(self, block_id: BlockId) -> bool:
        return block_id in self._cpu_all_block_indices

    def is_gpu_block(self, block_id: BlockId) -> bool:
        return block_id in self._gpu_all_block_indices

    def allocate_gpu_blocks_for_all_layers(
        self, seq_id: SeqId, num_blocks: int
    ) -> None:
        with self._lock:
            for layer in range(self.num_attn_layers):
                if seq_id not in self.layer_block_tables[layer]:
                    self.layer_block_tables[layer][seq_id] = BlockTable(self.block_size)
                table = self.layer_block_tables[layer][seq_id]
                for _ in range(num_blocks):
                    block = self._allocate_gpu_block()
                    table.blocks.append(block)
                    self.gid_to_seq.update(new_gid=block.block_id, seq_id=seq_id)

    def free_all_blocks_for_seq(self, seq_id: SeqId) -> None:
        """
        Free all blocks associated with a sequence ID across all layers.
        :param seq_id: Sequence ID to free blocks for.
        """
        with self._lock:
            for layer in range(self.num_attn_layers):
                if seq_id in self.layer_block_tables[layer]:
                    table = self.layer_block_tables[layer][seq_id]
                    for block in table.blocks:
                        self.gid_to_seq.update(old_gid=block.block_id)
                        self.free_block(block)
                        if self.is_gpu_block(block.block_id):
                            self._free_gpu_block(block)
                        elif self.is_cpu_block(block.block_id):
                            self._free_cpu_block(block)
                    del self.layer_block_tables[layer][seq_id]

    def allocate_gpu_blocks_for_layer(
        self, seq_id: SeqId, num_blocks: int, layer: int
    ) -> None:
        with self._lock:
            table = self.layer_block_tables[layer][seq_id]
            for _ in range(num_blocks):
                block = self._allocate_gpu_block()
                table.blocks.append(block)
                self.gid_to_seq.update(new_gid=block.block_id, seq_id=seq_id)
