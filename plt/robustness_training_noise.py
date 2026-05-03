import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()

lr = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]

for index, i in enumerate(lr):
    acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_lr{i}.pth")
    plt.plot(acc_list['extract_acc_list'], color=color[index], label=f'lr={index+1}e-4')
plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/acc_lr.pdf", format="pdf")
plt.show()

for index, i in enumerate(lr):
    acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_lr{i}.pth")
    plt.plot(acc_list['extract_acc_bch_list'], color=color[index], label=f'lr={index + 1}e-4')
plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/acc_lr_bch.pdf", format="pdf")
plt.show()

acc_list = torch.load(f"../data/extract_acc_resnet18_mnist_lr0.0002.pth")
plt.plot(acc_list['extract_acc_list'], color=color[0], label=f'mnist')
acc_list = torch.load(f"../data/extract_acc_resnet18_fashionmnist_lr0.0002.pth")
plt.plot(acc_list['extract_acc_list'], color=color[1], label=f'fashionmnist')
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0002.pth")
plt.plot(acc_list['extract_acc_list'], color=color[2], label=f'cifar10')
plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/dataset_lr.pdf", format="pdf")
plt.show()

acc_list = torch.load(f"../data/extract_acc_resnet18_mnist_lr0.0002.pth")
plt.plot(acc_list['extract_acc_bch_list'], color=color[0], label=f'mnist')
acc_list = torch.load(f"../data/extract_acc_resnet18_fashionmnist_lr0.0002.pth")
plt.plot(acc_list['extract_acc_bch_list'], color=color[1], label=f'fashionmnist')
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0002.pth")
plt.plot(acc_list['extract_acc_bch_list'], color=color[2], label=f'cifar10')
plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/dataset_lr_bch.pdf", format="pdf")
plt.show()