import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})
line_width = 1.1
color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']
alpha = [8, 12, 16, 20, 24]



acc_list = torch.load(f"../data/extract_acc_ResNet18_cifar10_{16}.pth")
plt.plot(acc_list['extract_acc_list'], color=color[0], label='Proposed', linewidth=line_width)
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_without_noise.pth")
plt.plot(acc_list['extract_acc_list'], color=color[1], label='w/o noise layer', linewidth=line_width)
# acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_without_sort.pth")
# plt.plot(acc_list['extract_acc_list'], color=color[2], label='w/o sort')
acc_list = torch.load(f"../data/extract_acc_resnet18_cifar10_with_mse.pth")
plt.plot(acc_list['extract_acc_list'], color=color[2], label='w/ MSE', linewidth=line_width)

plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/ablation_acc.pdf", format="pdf")
plt.show()


loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_{16}.pth")
plt.plot(loss_list, color=color[0], label='Proposed', linewidth=line_width)
loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_without_noise.pth")
plt.plot(loss_list, color=color[1], label='w/o noise layer', linewidth=line_width)
# loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_without_sort.pth")
# plt.plot(loss_list, color=color[2], label='w/o sort')
loss_list = torch.load(f"../data/loss_ResNet18_cifar10_with_secret_with_mse.pth")
plt.plot(loss_list, color=color[2], label='w/ MSE', linewidth=line_width)



plt.xlabel("Training epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/ablation_loss.pdf", format="pdf")
plt.show()



