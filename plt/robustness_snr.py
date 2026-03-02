import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18.5})
color = ['#E64A4A', '#5B9BD5', '#40A877', '#F0C078', '#8C8C8C', '#FF9900']
line_width = 1.1
data_dict = torch.load(f"../data/err_list_2_16.pth")

snr_list = data_dict["snr"]
err_list_2 = data_dict["acc"]
err_bch_list_2 = data_dict["acc_bch"]
err_list_1 = torch.load(f"../data/err_list_1_64.pth")


plt.plot(snr_list, err_list_1, color=color[2], label="Xu et al. [24]", linewidth=line_width)
plt.plot(snr_list, err_list_2, color=color[1], label="Proposed", linewidth=line_width)
plt.xlabel("SNR(dB)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig('../fig/snr.pdf', dpi=None, format='pdf')
plt.show()

plt.plot(snr_list, err_bch_list_2, color=color[1], label="w/ BCH", linewidth=line_width)
plt.plot(snr_list, err_list_2, color=color[0], label="w/o BCH", linewidth=line_width)

plt.xlabel("SNR(dB)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
plt.savefig('../fig/snr_bch.pdf', dpi=None, format='pdf')
plt.show()
