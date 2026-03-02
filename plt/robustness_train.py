import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})

color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']
line_width = 1.1
acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001.pth")
acc_list2 = torch.load(f"../project2/extract_acc_ResNet18_cifar10_0.0001_project2.pth")
plt.plot(acc_list2['extract_acc'], color=color[2], label='Xu et al. [24]', linewidth=line_width)
plt.plot(acc_list1['extract_acc_list'], color=color[1], label='Proposed', linewidth=line_width)

plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/train_extract_acc_1e-4.pdf", format="pdf")
plt.show()

acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0002.pth")
acc_list2 = torch.load(f"../project2/extract_acc_ResNet18_cifar10_0.0002_project2.pth")
plt.plot(acc_list2['extract_acc'], color=color[2], label='Xu et al. [24]', linewidth=line_width)
plt.plot(acc_list1['extract_acc_list'], color=color[1], label='Proposed', linewidth=line_width)

plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/train_extract_acc_2e-4.pdf", format="pdf")
plt.show()
