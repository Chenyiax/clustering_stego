import importlib
import math

import torch
import numpy as np
from torchvision import models

from clustering_stego import ClusteringStego
from utils.init_function import init_resnet18

# 设置设备 (CUDA or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 动态导入模块
module = importlib.import_module("get_data")
# 使用 getattr 获取函数对象
func = getattr(module, "get_"+ "cifar10" +"_data")
# 加载数据集
train_loader, test_loader = func()

# 创建载体网络模型
model_class = getattr(models, "resnet18")
model = model_class(weights=None)

model.load_state_dict(torch.load(f'../model/resnet18_cifar10_with_secret.pth'))
# 读取保存的数据（注意替换为实际的文件路径）
file_path = f'../data/secret_resnet18_cifar10.pth'
secret_bits = torch.load(file_path)['secret_bits']
init_func = init_resnet18
cs = ClusteringStego(init_func)

# 提取其中的秘密比特和矩阵

print(model)
model.to(device)
# 嵌入秘密信息的层
watermarklayer = model.layer2[0].conv2

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

    outputs = cs.decode(model)
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
torch.save({'err_list': err_list, 'snr_list': snr_list}, f'data/err_snr_xu_resnet18.pth')
