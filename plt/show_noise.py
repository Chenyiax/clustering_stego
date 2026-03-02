import math

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import savemat
from matplotlib import rc


from matplotlib.ticker import MaxNLocator, FuncFormatter
from torchvision import models

from utils.function import get_model_params, to_hist_tensor

rc('font', family='serif')  # 设置字体族为 serif
rc('font', serif='Times New Roman')  # 设置字体为 Times New Roman
rc('axes', unicode_minus=False)  # 解决负号显示问题
rc('font', size=18)  # 设置字体大小
# 设置公式字体
color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']

alexnet = models.alexnet()
alexnet.load_state_dict(torch.load(f"../model/alexnet_init_original.pth"))
params_init1 = torch.concatenate(get_model_params(alexnet))
alexnet.load_state_dict(torch.load(f"../model/alexnet_cifar10_without_secret.pth"))
params_init1 = torch.concatenate(get_model_params(alexnet))



bins = 1000
j = 0
hist_tensor_list = []
bin_center_list = []
for i in inacc_list:
    hist_tensor, bin_center = to_hist_tensor(i[-1], bins, range=(-0.2,0.2))
    hist_tensor_list.append(hist_tensor)
    bin_center_list.append(bin_center)

plt.plot(bin_center_list[0], hist_tensor_list[0], color=color[2], label='1e-4')
plt.plot(bin_center_list[1], hist_tensor_list[1], color=color[1], label='5e-5')
plt.plot(bin_center_list[2], hist_tensor_list[2], color=color[0], label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Difference of parameter values')
plt.ylabel('Frequency')

plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.ylim(top=0.051)

plt.legend()
plt.tight_layout()
plt.savefig('../pdf/noise_dis.pdf', dpi=None, format='pdf')
plt.show()

