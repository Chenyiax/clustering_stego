import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()
line_width = 1.1

acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001.pth")
acc_list2 = torch.load(f"../data/extract_acc_ResNet18_cifar10_0.0001_project2.pth")
plt.plot(acc_list2['extract_acc'], color=color[1], label='第三章方法', linewidth=line_width)
plt.plot(acc_list1['extract_acc_list'], color=color[0], label='第四章方法', linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/train_extract_acc_1e-4.png", format="png")
plt.show()

acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0002.pth")
acc_list2 = torch.load(f"../data/extract_acc_ResNet18_cifar10_0.0002_project2.pth")
plt.plot(acc_list2['extract_acc'], color=color[1], label='第三章方法', linewidth=line_width)
plt.plot(acc_list1['extract_acc_list'], color=color[0], label='第四章方法', linewidth=line_width)

plt.xlabel("训练轮数")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../png/train_extract_acc_2e-4.png", format="png")
plt.show()
