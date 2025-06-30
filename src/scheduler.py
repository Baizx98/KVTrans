def schedule_batch(batch):
    """
    简单的调度器，模拟生成的逻辑
    :param batch: 输入批次
    :return: 调度后的批次
    """
    # 这里可以添加调度逻辑，比如根据优先级、资源等进行排序
    # 目前简单返回原始批次
    # 要为batch分配block，包括prefill阶段和decode阶段，为每个层分配block TODO
    return batch


def cleanup_batch(batch):
    """
    清理批次中的已完成序列
    :param batch: 输入批次
    :return: 清理后的批次
    """
    return [seq for seq in batch if not seq.done]
