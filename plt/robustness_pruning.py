import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

data = torch.load("../data/random_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
model_acc_list = data['model_acc_list']

data = torch.load("../data/random_pruning_xu.pth")
acc_list_bch_xu = data['acc_list_bch']

acc_list_10 = torch.load("../data/random_pruning_acc_resnet18_10.pth")
acc_list_0 = torch.load("../data/random_pruning_acc_resnet18_0.pth")
acc_list_hao = torch.load("../data/random_pruning_acc_resnet18_hao.pth")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})

color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']
line_width = 1.1

prune_rates = np.linspace(0, 0.99, 100)
plt.plot(prune_rates, acc_list_0, color=color[5], label="Yang et al. [20]", linewidth=line_width)
plt.plot(prune_rates, acc_list_10, color=color[4], label="Yang et al. [21]", linewidth=line_width)
plt.plot(prune_rates, acc_list_hao, color=color[3], label="Hao et al. [23]", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch_xu, color=color[2], label="Xu et al. [24]", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch, color=color[1], label="Proposed", linewidth=line_width)

plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/pruning_random.pdf", format="pdf")
plt.show()

# prune_rates = np.linspace(0, 1, 100)
# plt.plot(prune_rates, model_acc_list, color=color[2])
# plt.xlabel("Pruning rate")
# plt.ylabel(r"Classification accuracy")
# plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
# plt.tight_layout()
# plt.savefig("../fig/pruning_acc_random.pdf", format="pdf")
# plt.show()

data = torch.load("../data/less_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
data = torch.load("../data/less_pruning_xu.pth")
acc_list = data['acc_list']
acc_list_bch_xu = data['acc_list_bch']
acc_list_10 = torch.load("../data/less_pruning_acc_resnet18_10.pth")
acc_list_0 = torch.load("../data/less_pruning_acc_resnet18_0.pth")
acc_list_hao = torch.load("../data/less_pruning_acc_resnet18_hao.pth")

plt.plot(prune_rates, acc_list_0, color=color[5], label="Yang et al. [20]", linewidth=line_width)
plt.plot(prune_rates, acc_list_10, color=color[4], label="Yang et al. [21]", linewidth=line_width)
plt.plot(prune_rates, acc_list_hao, color=color[3], label="Hao et al. [23]", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch_xu, color=color[2], label="Xu et al. [24]", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch, color=color[1], label="Proposed", linewidth=line_width)
plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig("../fig/pruning_less.pdf", format="pdf")
plt.show()

# prune_rates = np.linspace(0, 1, 100)
# plt.plot(prune_rates, model_acc_list, color=color[2])
# plt.xlabel("Pruning rate")
# plt.ylabel(r"Classification accuracy")
# plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
# plt.tight_layout()
# plt.savefig("../fig/pruning_acc_less.pdf", format="pdf")
# plt.show()

