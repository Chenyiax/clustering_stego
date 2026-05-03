import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plt.mpl_config import set_style

color = set_style()

line_width = 1.1
data_dict = torch.load(f"../data/err_list_2_16.pth")

snr_list = data_dict["snr"]
err_list_2 = data_dict["acc"]
err_bch_list_2 = data_dict["acc_bch"]
err_list_1 = torch.load(f"../data/err_list_1_64.pth")


plt.plot(snr_list, err_list_1, color=color[1], label="第三章方法", linewidth=line_width)
plt.plot(snr_list, err_list_2, color=color[0], label="第四章方法", linewidth=line_width)
plt.xlabel("信噪比")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig('../png/snr.png', dpi=None, format='png')
plt.show()

plt.plot(snr_list, err_bch_list_2, color=color[1], label="w/ BCH", linewidth=line_width)
plt.plot(snr_list, err_list_2, color=color[0], label="w/o BCH", linewidth=line_width)

plt.xlabel("信噪比")
plt.ylabel("提取准确率")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig('../png/snr_bch.png', dpi=None, format='png')
plt.show()
