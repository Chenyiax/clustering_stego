import argparse
import importlib
import math

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
from utils.standardizer import TensorStandardizer

# 设置设备 (CUDA or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="vgg16", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121')
parser.add_argument("--capacity", default=2000, type=int)
parser.add_argument("--alpha", default=5000, type=int, help='论文里没说具体的取值,这里暂且取2000')
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
        kernel_means = m.mean(dim=(2, 3)).view(-1)
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
# 创建载体网络模型
model_class = getattr(models, args.model)
model = model_class(weights=None)

model.load_state_dict(torch.load(f'../model/{args.model}_{args.dataset}_with_secret_{args.beta}.pth'))
# 读取保存的数据（注意替换为实际的文件路径）
file_path = f'../data/secret_{args.model}_{args.dataset}_{args.beta}.pth'
loaded_data = torch.load(file_path)

# 提取其中的秘密比特和矩阵
secret_bits = loaded_data['secret_bits']
watermarking_module.matrix = loaded_data['matrix']

print(model)
model.to(device)
# 嵌入秘密信息的层
if args.model == 'resnet18':
    watermarklayer = model.layer2[0].conv2
elif args.model == 'vgg16':
    watermarklayer = model.features[12]
elif args.model =='alexnet':
    watermarklayer = model.features[3]
elif args.model == 'densenet121':
    watermarklayer = model.features[5].conv
else:
    watermarklayer = None

secret_var = torch.var(watermarklayer.weight)
secret_var_np = secret_var.cpu().detach().numpy()  # 转换为numpy数组用于计算

# 生成-10到10之间的30个SNR值（单位：dB）
snr_db_values = np.linspace(-10, 10, 30)

# 根据SNR公式计算对应的噪声方差
# SNR(dB) = 10×log10(secret_var / noise_var) → noise_var = secret_var / 10^(SNR/10)
noise_var_arr = secret_var_np / (10 **(snr_db_values / 10))
err_list = []
snr_list = []
# 为每个噪声方差生成对应的噪声
for noise_var in noise_var_arr:
    # 生成均值为0、标准差为sqrt(noise_var)的噪声，与水印参数同形状
    noise = torch.normal(0, math.sqrt(noise_var), (watermarklayer.weight.size())).to(device)

    # 对生成的参数添加噪声()
    stego_params = watermarklayer.weight + noise

    outputs = watermarking_module.extract_watermark(stego_params)
    outputs = (outputs > 0.5).float()
    wrong = (outputs != secret_bits.to(device)).sum().item()
    err = 1 - wrong / secret_bits.numel()

    secret_var_cpu = secret_var.cpu().detach().numpy()

    # 计算信噪比
    snr = 10 * np.log(secret_var_cpu / noise_var) / np.log(10)
    err_list.append(err)
    snr_list.append(snr)

print(err_list)
print(snr_list)
torch.save({'err_list': err_list, 'snr_list': snr_list}, f'data/err_snr_yang_{args.model}_{args.beta}.pth')
