import random

import torch


class CustomRandomGenerator:
    def __init__(self, seed):
        self.seed = seed

    def randperm(self, n:int) -> torch.Tensor:
        """生成从0到n-1的乱序索引"""
        random.seed(self.seed)
        return torch.tensor(random.sample(range(n), n))