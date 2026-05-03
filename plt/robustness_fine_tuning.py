import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

plt.rcParams['font.family']=' Times New Roman, SimSun'# 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['font.size'] = 18

color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']

fine_tuning_rate = '0.0001'
line_width = 1.1
acc_list = torch.load(f"../data/acc_fine-tuning_nocross_1e-4.pth")
acc_list_bch = torch.load(f"../data/acc_bch_fine-tuning_nocross_1e-4.pth")
acc_list_xu = [1] * 100
acc_list_hao = [1] * 100
acc_list_10 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_10.pth")
acc_list_0 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_0.pth")


prune_rates = np.linspace(0, 100, 100)

plt.plot(prune_rates, acc_list_0, color=color[5], label="Yang 等人 [18]", linewidth=line_width)
plt.plot(prune_rates, acc_list_10, color=color[4], label="Yang 等人 [19]", linewidth=line_width)
plt.plot(prune_rates, acc_list_hao, color=color[3], label="Hao 等人 [21]", linewidth=line_width)
plt.plot(prune_rates, acc_list_xu, color=color[2], label="第三章方法", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch, color=color[1], label="第四章方法", linewidth=line_width)

plt.xlabel("微调轮数")
plt.ylabel("提取准确率")
plt.legend(fontsize=16)
# plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig(f'../png/fine-tuning_nocross_1e-4.png', format='png')
plt.show()

fine_tuning_rate = '5e-05'

acc_list = torch.load(f"../data/acc_fine-tuning_nocross_5e-5.pth")
acc_list_bch = torch.load(f"../data/acc_bch_fine-tuning_nocross_5e-5.pth")
acc_list_xu = [1] * 100
acc_list_hao = [1] * 100
acc_list_10 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_10.pth")
acc_list_0 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_0.pth")


prune_rates = np.linspace(0, 100, 100)

plt.plot(prune_rates, acc_list_0, color=color[5], label="Yang 等人 [18]", linewidth=line_width)
plt.plot(prune_rates, acc_list_10, color=color[4], label="Yang 等人 [19]", linewidth=line_width)
plt.plot(prune_rates, acc_list_hao, color=color[3], label="Hao 等人 [21]", linewidth=line_width)
plt.plot(prune_rates, acc_list_xu, color=color[2], label="第三章方法", linewidth=line_width)
plt.plot(prune_rates, acc_list_bch, color=color[1], label="第四章方法", linewidth=line_width)

plt.xlabel("微调轮数")
plt.ylabel("提取准确率")
plt.legend(fontsize=16)
plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig(f'../png/fine-tuning_nocross_{fine_tuning_rate}.png', format='png')
plt.show()
# acc_list = torch.load(f"../data/acc_fine-tuning_cross_{fine_tuning_rate}.pth")
# acc_list_bch = torch.load(f"../data/acc_bch_fine-tuning_cross_{fine_tuning_rate}.pth")
# acc_list_xu = torch.ones_like(acc_list)
# acc_list_hao = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_hao_cross.pth")
# acc_list_10 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_10_cross.pth")
# acc_list_0 = torch.load(f"../data/fine-tuning_acc_resnet18_cifar10_{fine_tuning_rate}_0_cross.pth")

# prune_rates = np.linspace(0, 100, 100)
# plt.plot(prune_rates, acc_list, color=color[1], label="w/o BCH")
# plt.plot(prune_rates, acc_list_bch, color=color[0], label="w/ BCH")
# plt.xlabel("Fine-tuning epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.ylim(0.95, 1.003)
# plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
# plt.tight_layout()
# plt.savefig(f'../fig/fine-tuning_{fine_tuning_rate}.pdf', format='pdf')
# plt.show()

