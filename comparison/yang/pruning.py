import argparse
import importlib

import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# 设置设备 (CUDA or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="resnet18", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121')
parser.add_argument("--capacity", default=5000, type=int)
parser.add_argument("--alpha", default=2000, type=int, help='论文里没说具体的取值,这里暂且取2000')
parser.add_argument("--dataset", default='mnist', type=str)
parser.add_argument("--beta", default=10, type=int, help='论文里没说具体的取值,这里暂且取10, 如果是22年的论文, 这个值取0')
args = parser.parse_args()
# 定义水印嵌入与提取模块
class WatermarkingModule:
    def __init__(self, capacity, beta, alpha):
        self.capacity = capacity
        self.beta = beta
        self.alpha = alpha
        self.secret_bits = torch.tensor(np.rint(np.random.rand(1, capacity)), dtype=torch.float32)
        self.matrix = None

    def generate_watermark(self):
        return self.secret_bits

    def extract_watermark(self, m: nn.Module):
        kernel_means = m.weight.mean(dim=(2, 3)).view(-1)
        if self.matrix == None:
            self.matrix = torch.randint(0, 2, (kernel_means.numel(), self.capacity), dtype=torch.float32).to(device)

        decoder_data_output = torch.matmul(kernel_means, self.matrix)  # 矩阵乘法
        decoder_data = torch.sigmoid(decoder_data_output)  # Sigmoid 激活函数

        return decoder_data.view(-1)


# 动态导入模块
module = importlib.import_module("get_data")
# 使用 getattr 获取函数对象
func = getattr(module, "get_"+args.dataset+"_data")
# 加载数据集
train_loader, test_loader = func()

# 水印方法
watermarking_module = WatermarkingModule(capacity=args.capacity, alpha=args.alpha, beta=args.beta)
data = torch.load(f"data/secret_{args.model}_{args.dataset}_{args.beta}.pth")
watermarking_module.matrix = data['matrix']
watermarking_module.secret_bits = data['secret_bits']
# 创建载体网络模型
model_class = getattr(models, args.model)
model = model_class(weights=None)

model.load_state_dict(torch.load(f"model/{args.model}_{args.dataset}_with_secret_{args.beta}.pth"))

model.to(device)
print(model)
# 定义损失函数与优化器

# 嵌入秘密信息的层
if args.model == 'resnet18':
    watermarklayer = model.layer2[0].conv2
elif args.model == 'vgg16':
    watermarklayer = model.features[12]
elif args.model =='alexnet':
    watermarklayer = model.features[3]
elif args.model == 'densenet121':
    watermarklayer = model.features[10].denselayer16.conv1
else:
    watermarklayer = None

# 训练模型
num_epochs = 100

watermark = watermarking_module.generate_watermark().to(device)
extract_watermark = watermarking_module.extract_watermark(watermarklayer)
predictions = (extract_watermark > 0.5).float()
correct = (watermark == predictions).sum().item()
accuracy = correct / watermark.numel()
print(accuracy)

acc_list = []
for epoch in range(num_epochs):
    prune_ratio = 0.01 * epoch

    weight = watermarklayer.weight
    total_weights = weight.size(0) * weight.size(1) * weight.size(2) * weight.size(3)
    num_prune = int(total_weights * prune_ratio)
    flat_weight = weight.flatten().to("cpu").detach().numpy()

    # 基于权重绝对值的大小排序
    sorted_indices = np.argsort(np.abs(flat_weight))
    prune_indices = sorted_indices[:num_prune]

    # 随机选择要剪枝的索引
    # prune_indices = np.random.choice(total_weights, num_prune, replace=False)

    flat_weight[prune_indices] = 0
    weight = torch.tensor(flat_weight).view(weight.shape).to(device)
    watermarklayer.weight = nn.Parameter(weight)

    extract_watermark = watermarking_module.extract_watermark(watermarklayer)

    predictions = (extract_watermark > 0.5).float()
    correct = (watermark == predictions).sum().item()
    accuracy = correct / watermark.numel()
    acc_list.append(accuracy)
    print("Epoch:", epoch, "Extraction Accuracy of Secret Information:", accuracy)


torch.save(acc_list, f"data/less_pruning_acc_{args.model}_{args.beta}.pth")
