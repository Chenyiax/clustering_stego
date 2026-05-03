import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()

line_width = 1.1
acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001_kaimin.pth")
acc_list2 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001_xavier_uniform.pth")
acc_list3 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001.pth")

plt.plot(acc_list1['extract_acc_list'], color=color[0], label='Kaiming 初始化', linewidth=line_width)
plt.plot(acc_list2['extract_acc_list'], color=color[1], label='Xavier 初始化', linewidth=line_width)
plt.plot(acc_list3['extract_acc_list'], color=color[2], label='默认初始化', linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/ablation_init_acc.png", format="png")
plt.show()

loss_list1 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001_kaimin.pth")
loss_list2 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001_xavier_uniform.pth")
loss_list3 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001.pth")


plt.plot(loss_list1, color=color[0], label='Kaiming 初始化', linewidth=line_width)
plt.plot(loss_list2, color=color[1], label='Xavier 初始化', linewidth=line_width)
plt.plot(loss_list3, color=color[2], label='默认初始化', linewidth=line_width)
plt.xlabel("训练轮数")
plt.ylabel("损失")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/ablation_init_loss.png", format="png")
plt.show()
