import torch
from torchvision import models

from utils.function import calculate_entropy_with_hist, calculate_kl_divergence
from utils.random_generator import CustomRandomGenerator

resnet18 = models.resnet18(pretrained=False)
random_generator = CustomRandomGenerator(42)

with torch.no_grad():
    for name, m in resnet18.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            param = m.weight.flatten()
            indices = random_generator.randperm(param.size(0))
            var = param.var().item()
            if var < 2e-4:
                continue
            # 置乱 tensor
            param_random = param[indices]
            entropy_random = calculate_entropy_with_hist(param_random)
            entropy_normal = calculate_entropy_with_hist(param)

            kl = calculate_kl_divergence(param, param_random)
            print(f"{name} entropy_difference: {torch.abs(entropy_random - entropy_normal)}, kl:{kl}")

