import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import time
from torchvision import models

from tqdm import tqdm

from get_data import get_cifar10_data, get_mnist_data, get_fashionmnist_data

# 设置设备 (CUDA or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="resnet18", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121')
parser.add_argument("--capacity", default=5000, type=int)
parser.add_argument("--alpha", default=2000, type=int, help='论文里没说具体的取值,这里暂且取2000')
parser.add_argument("--dataset", default='mnist', type=str)
parser.add_argument("--beta", default=10, type=int, help='论文里没说具体的取值,这里暂且取10, 如果是22年的论文, 这个值取0')
parser.add_argument("--quantize", default='bool', type=str)
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

weight = watermarklayer.weight
mean = torch.mean(weight)
if args.quantize == 'bool':
    mean = torch.mean(weight)
    # 计算数据分布范围（最大值与最小值的差）
    data_range = torch.max(weight) - torch.min(weight)
    if data_range == 0:
        data_range = 1e-8  # 避免除以0

    # 二值化：大于均值为1，否则为0（基于中点）
    binary_mask = (weight > mean).float()

    # 基于中点还原：将0/1映射回原始数据范围
    # 0对应均值以下的范围，1对应均值以上的范围
    binary_weight = mean + (binary_mask - 0.5) * data_range
elif args.quantize == 'int8':
    # 8位整数量化：以均值为中点，映射到[-127, 127]范围
    mean = torch.mean(weight)
    # 先减去均值，将分布中心移到0点
    centered_weight = weight - mean
    # 找到最大值用于归一化
    max_val = torch.max(torch.abs(centered_weight))
    if max_val == 0:
        max_val = 1e-8  # 避免除以0
    # 归一化到[-1, 1]范围，量化后再恢复偏移
    binary_weight = (torch.round(centered_weight / max_val * 127) / 127 * max_val) + mean
elif args.quantize == 'int4':
    # 4位整数量化：以均值为中点，映射到[-15, 15]范围
    mean = torch.mean(weight)
    centered_weight = weight - mean
    max_val = torch.max(torch.abs(centered_weight))
    if max_val == 0:
        max_val = 1e-8
    binary_weight = (torch.round(centered_weight / max_val * 15) / 15 * max_val) + mean
elif args.quantize == 'int16':
    # 16位整数量化：以均值为中点，映射到[-32767, 32767]范围
    mean = torch.mean(weight)
    centered_weight = weight - mean
    max_val = torch.max(torch.abs(centered_weight))
    if max_val == 0:
        max_val = 1e-8
    binary_weight = (torch.round(centered_weight / max_val * 32767) / 32767 * max_val) + mean
elif args.quantize == 'float16':
    # 16位浮点数量化：通过类型转换实现
    binary_weight = weight.to(torch.float16).to(torch.float32)
elif args.quantize == 'int32':
    # 32位整数量化：以均值为中点，映射到[-2147483647, 2147483647]范围
    mean = torch.mean(weight)
    centered_weight = weight - mean
    max_val = torch.max(torch.abs(centered_weight))
    if max_val == 0:
        max_val = 1e-8
    binary_weight = (torch.round(centered_weight / max_val * 2147483647) / 2147483647 * max_val) + mean
elif args.quantize == 'int2':
    # 2位整数量化：以均值为中点，映射到[-3, 3]范围
    mean = torch.mean(weight)
    centered_weight = weight - mean
    max_val = torch.max(torch.abs(centered_weight))
    if max_val == 0:
        max_val = 1e-8
    binary_weight = (torch.round(centered_weight / max_val * 3) / 3 * max_val) + mean


watermarklayer.weight = nn.Parameter(binary_weight)

extract_watermark = watermarking_module.extract_watermark(watermarklayer)

predictions = (extract_watermark > 0.5).float()
correct = (watermark == predictions).sum().item()
accuracy = correct / watermark.numel()
acc_list.append(accuracy)
print(accuracy)
