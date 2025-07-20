from typing import List
import random


class Sequence:
    def __init__(self, seq_id, seq_len, done=False):
        self.seq_id = seq_id
        self.seq_len = seq_len
        self.done = done
        self.block_num = 0

    def mark_down(self):
        """Mark the sequence as done."""
        self.done = True

    def generate_new_token(self):
        self.seq_len += 1

        # 生成终止条件
        if self.seq_len >= 10:
            flag = random.random()
            if flag < 0.9:  # 50% 的概率终止
                self.mark_down()


def generate_random_batch(batch_size: int, seq_length: int) -> List[Sequence]:
    """
    Generate a random batch of sequences.
    :param batch_size: Number of sequences in the batch.
    :param seq_length: Length of each sequence.
    :return: List of Sequence objects.
    """
    return [Sequence(seq_id=i, seq_len=seq_length) for i in range(batch_size)]
