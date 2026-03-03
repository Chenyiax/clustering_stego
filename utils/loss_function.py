import torch
from torch import nn


# 熵正则化损失
class EntropyRegularizedLoss(nn.Module):
    def __init__(self, entropy_weight):
        super(EntropyRegularizedLoss, self).__init__()
        self.entropy_weight = entropy_weight

    def forward(self, outputs):
        # 计算熵
        epsilon = 1e-10
        outputs = torch.clamp(outputs, epsilon, 1.0 - epsilon)
        entropy = (outputs * torch.log(outputs) + (1 - outputs) * torch.log(1 - outputs)).mean()

        # 正则化损失
        loss = self.entropy_weight * entropy
        return loss