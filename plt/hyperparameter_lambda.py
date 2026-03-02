import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})
line_width = 1.1

color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']

acc_list1 = torch.load(f"../data/extract_acc_resnet18_cifar10_var5e-05.pth")
acc_list2 = torch.load(f"../data/extract_acc_resnet18_cifar10_var0.0005.pth")
acc_list3 = torch.load(f"../data/extract_acc_resnet18_cifar10_lr0.0001.pth")

plt.plot(acc_list1['extract_acc_list'], color=color[0], label=r'$\lambda$=5e-5', linewidth=line_width)
plt.plot(acc_list2['extract_acc_list'], color=color[1], label=r'$\lambda$=5e-4', linewidth=line_width)
plt.plot(acc_list3['extract_acc_list'], color=color[2], label=r'$\lambda$=2e-4', linewidth=line_width)

plt.xlabel("Training epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/hyperparameter_lambda_acc.pdf", format="pdf")
plt.show()

loss_list1 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_var5e-05.pth")
loss_list2 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_var0.0005.pth")
loss_list3 = torch.load(f"../data/loss_resnet18_cifar10_with_secret_lr0.0001.pth")


plt.plot(loss_list1, color=color[0], label=r'$\lambda$=5e-5', linewidth=line_width)
plt.plot(loss_list2, color=color[1], label=r'$\lambda$=5e-4', linewidth=line_width)
plt.plot(loss_list3, color=color[2], label=r'$\lambda$=2e-4', linewidth=line_width)
plt.xlabel("Training epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/hyperparameter_lambda_loss.pdf", format="pdf")
plt.show()
