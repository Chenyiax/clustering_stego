'''
文件名: kl_diver.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制 kl 散度拟合程度
'''
import argparse
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

import init_function
from classifier_model import LSTM
from utils.function import to_hist_tensor, calculate_entropy_with_hist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="densenet121", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121, lstm')
parser.add_argument('--dataset', default="cifar10", type=str, help='数据集, 可选:mnist, fashionmnist, cifar10, sst2')
args = parser.parse_args()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})

color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']

if args.model == "lstm" or args.model == "transformer":
    init_func = init_function.init_nlp
else:
    init_func = getattr(init_function, "init_"+args.model)

target_var = 2e-4

dataset_name = args.dataset

if args.model == "lstm":
    model1 = LSTM()
    model2 = LSTM()
else:
    model1 = getattr(models, args.model)(weights=None)
    model2 = getattr(models, args.model)(weights=None)

# model1.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_with_secret.pth"))
# model1.load_state_dict(torch.load(f"../project2/AlexNet_cifar10_with_secret_project2.pth"))
model1.load_state_dict(torch.load(f"../model/densenet121_cifar10_with_secret_var5e-05.pth"))
model2.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_without_secret.pth"))

# model1 = torch.load(f"../model/{model_name}_with_secret_init.pth")
# model2 = torch.load(f"../model/{model_name}_without_secret_init.pth")

params_with_secret = []
params_without_secret = []

with torch.no_grad():  # 禁用梯度计算
    for name, m in model1.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            weight_var, bias_var = init_func(m, name)

            # 统计这层模型参数个数
            if hasattr(m, 'bias') and m.bias is not None and bias_var > target_var:
                params_num = m.weight.numel() + m.bias.numel()
            else:
                params_num = m.weight.numel()

            # 如果参数过多则不生成参数
            if params_num < 1000:
                continue

            # 如果方差过小则不嵌入秘密信息
            if weight_var < target_var:
                continue
            params_with_secret.append(m.weight.detach().reshape(-1).to("cpu"))


with torch.no_grad():  # 禁用梯度计算
    for name, m in model2.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            weight_var, bias_var = init_func(m, name)

            # 统计这层模型参数个数
            if hasattr(m, 'bias') and m.bias is not None and bias_var > target_var:
                params_num = m.weight.numel() + m.bias.numel()
            else:
                params_num = m.weight.numel()

            # 如果参数过多则不生成参数
            if params_num < 1000:
                continue

            # 如果方差过小则不嵌入秘密信息
            if weight_var < target_var:
                continue

            params_without_secret.append(m.weight.detach().reshape(-1).to("cpu"))

num_params = len(params_with_secret)
k = 0
kl_list = []
entropy_list = []
for i, j in zip(params_without_secret, params_with_secret):
    bins = int(math.sqrt(len(i)))
    hist_tensor1, bin_centers1 = to_hist_tensor(i, bins)
    hist_tensor2, bin_centers2 = to_hist_tensor(j, bins)

    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
    kl_list.append(kl_divergence)
    entropy1 = calculate_entropy_with_hist(hist_tensor1)
    entropy2 = calculate_entropy_with_hist(hist_tensor2)
    entropy_list.append(torch.abs(entropy1 - entropy2))
    # print(kl_divergence)

    # plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
    # plt.plot(bin_centers1, hist_tensor1, color=color[1], label='Clean')
    # plt.plot(bin_centers2, hist_tensor2, color=color[0], label='Stego')
    # plt.xlabel('Parameter values')
    # plt.ylabel('Frequency')
    # plt.title(f"KL = {kl_divergence:.4f}")
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    # plt.legend()
    # plt.tight_layout()
    # plt.close()
    # k += 1

kl_list = torch.tensor(kl_list)
entropy_list = torch.tensor(entropy_list)

print(torch.mean(entropy_list))
print(torch.mean(kl_list))
