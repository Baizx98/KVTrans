""""""

from collections import deque

from typing import List, Dict, Optional, Deque, FrozenSet, Tuple
from config import CacheConfig, ModelConfig, DeviceConfig
from sequence import Sequence
import torch
import time
import threading

BlockId = int
SeqId = int


class Block:
    def __init__(self, block_id: BlockId):
        self.block_id = block_id


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
        self.gid_to_seq: Dict[BlockId, SeqId] = {}

        # 对block manager的操作们是否需要加锁呢？
        self._lock = threading.Lock()
        # 我要打问号了

    @property
    def buffer_blocks(self) -> int:
        return self._buffer_blocks

    @buffer_blocks.setter
    def buffer_blocks(self, value: int) -> None:
        if value < 0:
            raise ValueError("Buffer blocks cannot be negative")
        self._buffer_blocks = value

    def can_allocate_gpu_blocks(self, n: int) -> bool:
        # TODO watermark and buffer
        return len(self._gpu_free_block_indices) >= n

    def can_allocate_cpu_blocks(self, n: int) -> bool:
        # TODO watermark and buffer
        return len(self._cpu_free_block_indices) >= n

    def allocate_gpu_block_id(self) -> BlockId:
        return (
            self._gpu_free_block_indices.popleft()
            if self._gpu_free_block_indices
            else -1
        )

    def allocate_cpu_block_id(self) -> BlockId:
        return (
            self._cpu_free_block_indices.popleft()
            if self._cpu_free_block_indices
            else -1
        )

    def allocate_gpu_block(self) -> Block:
        block_id = self.allocate_gpu_block_id()
        if block_id == -1:
            raise RuntimeError("No available GPU blocks")
        # Block Pool
        return Block(block_id)

    def allocate_cpu_block(self) -> Block:
        block_id = self.allocate_cpu_block_id()
        if block_id == -1:
            raise RuntimeError("No available CPU blocks")
        return Block(block_id)

    def free_gpu_block(self, block: Block):
        gid = block.block_id
        self._gpu_free_block_indices.append(gid)
        del block

    def free_gpu_block_id(self, block_id: BlockId):
        self._gpu_free_block_indices.append(block_id)

    def free_cpu_block(self, block: Block):
        gid = block.block_id
        self._cpu_free_block_indices.append(gid)
        del block

    def free_cpu_block_id(self, block_id: BlockId):
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

    def is_kv_cache_ready(self, batch: List[Sequence], layer: int) -> bool:
        """检查该batch的该层的所需最少kv块是否都在GPU上"""
        for seq in batch:
            if seq.seq_id not in self.layer_block_tables[layer]:
                return False
        # 先这样写，后面再改，加具体的逻辑
        return True

    def get_layer_blocks_by_importance(self, layer: int) -> List[Block]:
        # Dummy: importance = block_id
        # 应该从GPU块中挑选
        # TODO 对每个请求分别处理，应该接收
        # TODO 这里应该对请求也加一层筛选，来兼容perfix cache
        if layer >= self.num_attn_layers:
            return []
        all_blocks: List[Block] = []
        for table in self.layer_block_tables[layer].values():
            all_blocks.extend(table.blocks)
        all_blocks = [
            block for block in all_blocks if self.is_gpu_block(block.block_id)
        ]
        return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def predict_next_layer_needed_blocks(self, layer: int) -> List[Block]:
        # 应该从CPU块中挑选
        if layer >= self.num_attn_layers:
            return []
        all_blocks: List[Block] = []
        for table in self.layer_block_tables[layer].values():
            all_blocks.extend(table.blocks)
        all_blocks = [
            block for block in all_blocks if self.is_cpu_block(block.block_id)
        ]
        return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def get_offload_plan(
        self,
        blocks: List[Block],
    ) -> List[Tuple[int, int]]:
        # 问题是，块实例跟seq是绑定在一起的，我在释放块id的时候，是单独释放的id
        # 然而，再卸载完成后，我要更新seq中的block的id，而不是释放block这个对象
        with self._lock:
            physical_block_id_mapping: List[Tuple[int, int]] = []
            block_num = len(blocks)
            if self.can_allocate_cpu_blocks(block_num):
                for i in range(block_num):
                    gpu_block_id = blocks[i].block_id
                    cpu_block_id = self.allocate_cpu_block_id()
                    physical_gpu_id = self.get_device_and_pid(gpu_block_id)[1]
                    physical_cpu_id = self.get_device_and_pid(cpu_block_id)[1]
                    physical_block_id_mapping.append((physical_gpu_id, physical_cpu_id))
            else:
                raise RuntimeError("Not enough CPU blocks available for offloading")

            return physical_block_id_mapping

    def get_prefetch_plan(self, blocks: List[Block]) -> List[Tuple[int, int]]:
        with self._lock:
            physical_block_id_mapping: List[Tuple[int, int]] = []
            block_num = len(blocks)
            if self.can_allocate_gpu_blocks(block_num):
                for i in range(block_num):
                    cpu_block_id = blocks[i].block_id
                    gpu_block_id = self.allocate_gpu_block_id()
                    physical_cpu_id = self.get_device_and_pid(cpu_block_id)[1]
                    physical_gpu_id = self.get_device_and_pid(gpu_block_id)[1]
                    physical_block_id_mapping.append((physical_cpu_id, physical_gpu_id))
            else:
                raise RuntimeError("Not enough GPU blocks available for prefetching")

            return physical_block_id_mapping

    def get_transfer_plan(
        self, blocks: List[Block], src_device: torch.device, dst_device: torch.device
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError(
            "This method should be implemented in subclasses to provide specific transfer plans."
        )

    def update_block_device_offload(
        self, plan: List[tuple[int, int]], original_blocks: List[Block]
    ) -> None:
        """
        更新原始块的设备信息
        :param plan: [(gpu_block_id, cpu_block_id), ...]
        :param original_blocks: List[Block]
        """
        print("offload plan yes")
        with self._lock:
            for i, (gpu_pid, cpu_pid) in enumerate(plan):
                # NOTE plan里面是物理id，gid 里面是全局id
                gpu_gid = self.get_gid(torch.device("cuda"), gpu_pid)
                cpu_gid = self.get_gid(torch.device("cpu"), cpu_pid)
                original_blocks[i].block_id = cpu_gid
                try:
                    seq_id = self.gid_to_seq[gpu_gid]
                except KeyError:
                    print(
                        f"[Warning] gpu_gid {gpu_gid} 不在 gid_to_seq 中，跳过该次 offload"
                    )
                    continue
                self.gid_to_seq[cpu_gid] = seq_id
                del self.gid_to_seq[gpu_gid]
                self.free_gpu_block_id(gpu_gid)

    def update_block_device_prefetch(
        self, plan: List[tuple[int, int]], original_blocks: List[Block]
    ) -> None:
        with self._lock:
            for i, (cpu_pid, gpu_pid) in enumerate(plan):
                cpu_gid = self.get_gid(torch.device("cpu"), cpu_pid)
                gpu_gid = self.get_gid(torch.device("cuda"), gpu_pid)
                original_blocks[i].block_id = gpu_gid
                try:
                    seq_id = self.gid_to_seq[cpu_gid]
                except KeyError:
                    print(
                        f"[Warning] cpu_gid {cpu_gid} 不在 gid_to_seq 中，跳过该次 prefetch"
                    )
                    continue
                self.gid_to_seq[gpu_gid] = seq_id
                del self.gid_to_seq[cpu_gid]
                self.free_cpu_block_id(cpu_gid)

    def kv_cache_ready(self, batch: List[Sequence], layer: int) -> bool:
        for seq in batch:
            if seq.seq_id not in self.layer_block_tables[layer]:
                return False
        return True

    def wait_for_kv_cache_ready(self, batch: List[Sequence], layer: int) -> None:
        while not self.kv_cache_ready(batch, layer):
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
                    block = self.allocate_gpu_block()
                    table.blocks.append(block)
                    self.gid_to_seq[block.block_id] = seq_id

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
                        del self.gid_to_seq[block.block_id]
                        if self.is_gpu_block(block.block_id):
                            self.free_gpu_block(block)
                        elif self.is_cpu_block(block.block_id):
                            self.free_cpu_block(block)
                    del self.layer_block_tables[layer][seq_id]

    def allocate_gpu_blocks_for_layer(
        self, seq_id: SeqId, num_blocks: int, layer: int
    ) -> None:
        with self._lock:
            table = self.layer_block_tables[layer][seq_id]
            for _ in range(num_blocks):
                block = self.allocate_gpu_block()
                table.blocks.append(block)
                self.gid_to_seq[block.block_id] = seq_id

    def update_blocks_after_transfer(self) -> None:
        pass
