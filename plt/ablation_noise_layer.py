import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()

line_width = 1.1
alpha = [8, 12, 16, 20, 24]

acc_list = torch.load(f"../data/extract_acc_ResNet18_cifar10_{16}.pth")
plt.plot(acc_list['extract_acc_list'], color=color[0], label='本章方法', linewidth=line_width)
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_without_noise.pth")
plt.plot(acc_list['extract_acc_list'], color=color[1], label='无噪声层', linewidth=line_width)
# acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_without_sort.pth")
# plt.plot(acc_list['extract_acc_list'], color=color[2], label='w/o sort')
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_with_mse.pth")
plt.plot(acc_list['extract_acc_list'], color=color[2], label='无SMSE', linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/ablation_acc.png", format="png")
plt.show()


loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_{16}.pth")
plt.plot(loss_list, color=color[0], label='本章方法', linewidth=line_width)
loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_without_noise.pth")
plt.plot(loss_list, color=color[1], label='无噪声层', linewidth=line_width)
# loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_without_sort.pth")
# plt.plot(loss_list, color=color[2], label='w/o sort')
loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_with_mse.pth")
plt.plot(loss_list, color=color[2], label='无SMSE', linewidth=line_width)


plt.xlabel("训练轮数")
plt.ylabel("损失")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/ablation_loss.png", format="png")
plt.show()



