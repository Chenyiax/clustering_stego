import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})

color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']
line_width = 1.1
acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001_kaimin.pth")
acc_list2 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001_xavier_uniform.pth")
acc_list3 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001.pth")

plt.plot(acc_list1['extract_acc_list'], color=color[0], label='Kaiming initialization', linewidth=line_width)
plt.plot(acc_list2['extract_acc_list'], color=color[1], label='Xavier initialization', linewidth=line_width)
plt.plot(acc_list3['extract_acc_list'], color=color[2], label='Default', linewidth=line_width)

plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/ablation_init_acc.pdf", format="pdf")
plt.show()

loss_list1 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001_kaimin.pth")
loss_list2 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001_xavier_uniform.pth")
loss_list3 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001.pth")


plt.plot(loss_list1, color=color[0], label='Kaiming initialization', linewidth=line_width)
plt.plot(loss_list2, color=color[1], label='Xavier initialization', linewidth=line_width)
plt.plot(loss_list3, color=color[2], label='Default', linewidth=line_width)
plt.xlabel("Training epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/ablation_init_loss.pdf", format="pdf")
plt.show()
