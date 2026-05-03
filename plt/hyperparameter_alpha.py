import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()

alpha = [8, 12, 16, 20, 24]

line_width = 1.1

for i, a in enumerate(alpha):
    acc_list = torch.load(f"../data/extract_acc_ResNet18_cifar10_{a}.pth")
    plt.plot(acc_list['extract_acc_list'], color=color[i], label=f"$\\alpha$ = {a}", linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/alpha_acc.png", format="png")
plt.show()

for i, a in enumerate(alpha):
    loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_{a}.pth")
    plt.plot(loss_list, color=color[i], label=f"$\\alpha$ = {a}", linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("损失")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/alpha_loss.png", format="png")
plt.show()



