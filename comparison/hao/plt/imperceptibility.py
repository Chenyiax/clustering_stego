import argparse
import math

import torch
import torchvision.models as models
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from utils.function import to_hist_tensor, calculate_entropy_with_hist, calculate_kl_divergence

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="resnet18", type=str, help='模型, 可选:alexnet, vgg16, resnet18')
args = parser.parse_args()

model_class = getattr(models, args.model)
model1 = model_class(weights=None)
model2 = model_class(weights=None)
model1.load_state_dict(torch.load(f"../../../model/{args.model}_cifar10_with_secret_hao.pth"))
model2.load_state_dict(torch.load(f"../../../model/{args.model}_cifar10_without_secret.pth"))

if args.model == 'alexnet':
    watermarklayer = model1.features[0]
    originlayer = model2.features[0]
elif args.model == 'vgg16':
    watermarklayer = model1.features[0]
    originlayer = model2.features[0]
elif args.model == 'resnet18':
    watermarklayer = model1.conv1
    originlayer = model2.conv1
else:
    watermarklayer = model1.features[4]
    originlayer = model2.features[4]

# watermarklayer = model1.conv1
# originlayer = model2.conv1

# watermarklayer = model1.features[0]
# originlayer = model2.features[0]


bins = int(math.sqrt(len(originlayer.weight.view(-1))))
hist_tensor1, bin_centers1 = to_hist_tensor(originlayer.weight.view(-1), bins)
hist_tensor2, bin_centers2 = to_hist_tensor(watermarklayer.weight.view(-1), bins)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})

# 颜色数组
color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']

plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.plot(bin_centers1, hist_tensor1, color=color[1], label='Cover model')
plt.plot(bin_centers2, hist_tensor2, color=color[0], label='Stego model')
plt.xlabel('Parameter values')
plt.ylabel('Frequency')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.legend()
plt.tight_layout()
# plt.savefig(f'../fig/{args.model}_cifar10_kl_compare_{args.beta}.pdf', dpi=None, format='pdf')
plt.show()

kl_divergence = calculate_kl_divergence(originlayer.weight, watermarklayer.weight)
entropy1 = calculate_entropy_with_hist(originlayer.weight)
entropy2 = calculate_entropy_with_hist(watermarklayer.weight)

print(entropy1 - entropy2)
print(kl_divergence)