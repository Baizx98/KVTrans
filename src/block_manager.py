""""""

from collections import deque

from typing import List, Dict, Optional, Deque, FrozenSet, Tuple
from config import CacheConfig, ModelConfig, DeviceConfig

import torch


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

    def free_cpu_block(self, block: Block):
        gid = block.block_id
        self._cpu_free_block_indices.append(gid)
        del block

    def get_device_and_pid(self, gid: BlockId) -> Tuple[torch.device, int]:
        if gid >= 0 and gid < self.num_gpu_blocks:
            return torch.device("cuda"), gid
        elif (
            gid >= self.num_gpu_blocks
            and gid < self.num_gpu_blocks + self.num_cpu_blocks
        ):
            return torch.device("cpu"), gid - self.num_gpu_blocks
        raise ValueError(f"Invalid gid: {gid}")

    def get_layer_blocks_by_importance(self, layer: int) -> List[Block]:
        # Dummy: importance = block_id
        # 应该从GPU块中挑选
        # TODO 对每个请求分别处理，应该接收
        if layer >= self.num_attn_layers:
            return []
        all_blocks: List[Block] = []
        for table in self.layer_block_tables[layer].values():
            all_blocks.extend(table.blocks)
        return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def predict_next_layer_needed_blocks(self, layer: int) -> List[Block]:
        # 应该从CPU块中挑选
        if layer >= self.num_attn_layers:
            return []
        all_blocks: List[Block] = []
        for table in self.layer_block_tables[layer].values():
            all_blocks.extend(table.blocks)
        return sorted(all_blocks, key=lambda block: block.block_id, reverse=True)

    def get_offload_plan(
        self,
        blocks: List[Block],
    ) -> List[Tuple[int, int]]:
        # List[Tuple[int, int]]: List of tuples (src_block_id, tgt_block_id)
        # TODO 数据结构设计
        # 对所有块进行排序，对每个传输单位（若干块）就生成一个卸载计划 ×
        # 每个卸载计划里分配了目标设备的块号
        # 每个卸载计划一旦产生，必须传输完毕
        # 返回一个mapping即可？ 看看vllm中的worker input是如何设计的？
        # blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in, List[Tuple[int,int]]
        # blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        # blocks_to_copy=scheduler_outputs.blocks_to_copy,
        # blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
        #                                 device="cpu",
        #                                 dtype=torch.int64).view(-1, 2)
        # blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
        #                                  device="cpu",
        #                                  dtype=torch.int64).view(-1, 2)
        ## `blocks_to_copy` is a gpu tensor. The src and tgt of
        ## blocks to copy are in the same device, and `blocks_to_copy`
        ## can be used directly within cuda kernels.
        # blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
        #                              device=self.device,
        #                              dtype=torch.int64).view(-1, 2)
        # 必须先分配，再传输，最后释放
        # 应该返回tensor和list[Tuple[int, int]]
        # 那么在offload plan中，应该先申请CPU块号，返回映射tensor

        return [(1, 2)]

    def offload(self):
        sorted_important_block = self.get_layer_blocks_by_importance(0)
        # 生成卸载计划

    def get_prefetch_plan(self, layer: int):
        # TODO 预取计划
        return None

    # 重点是我需要一个独立的模块来完成块元数据管理和物理数据管理的整合或者通信
    # 每传输一个单位，就要执行 目标设备块号预分配 块传输 源设备块号释放 这一原子步骤
