"""
文件名: test_snr.py
作者: 徐辰屹
日期: 2024年5月4日

说明:
绘制模型抗噪新能曲线的文件
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import *
from utils import standardizer
from utils.function import get_secretbits_BCH, bch_decode, cut_tensor, get_secretbits
from utils.standardizer import TensorStandardizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = 16

secret_bits_encoder = torch.load(f"../model/encoder_{alpha}.pth").eval()
secret_bits_decoder = torch.load(f"../model/decoder_{alpha}.pth").eval()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 14})

color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']

params_num = 200000   # 生成参数个数
noise_var_arr = np.linspace(1e-2, 10, 30)  # 噪声方差
snr_list = []
err_list = []
err_bch_list = []

standardizer = TensorStandardizer()
with torch.no_grad():
    for noise_var in noise_var_arr:
        # 获取秘密信息
        secret_bits, secret_bits_bch = get_secretbits_BCH(params_num)
        secret_bits = secret_bits.to(device)

        origin_params = torch.normal(0, 1, size=(params_num,)).to(device)

        # 参数裁剪
        cutted_params, last_params = cut_tensor(origin_params, secret_bits.numel() * alpha)
        # 标准化
        cutted_params = standardizer.standardize(cutted_params).view(-1, alpha)

        stego_params = secret_bits_encoder(cutted_params, secret_bits)
        secret_var = stego_params.var().detach().cpu().numpy()

        noise = torch.normal(0, math.sqrt(noise_var), stego_params.size()).to(device)

        # 对生成的参数添加噪声()
        stego_params = stego_params + noise

        outputs = secret_bits_decoder(stego_params).to("cpu")
        outputs = (outputs > 0.5).float()

        outputs_bch = bch_decode(outputs.view(-1, 128).numpy()).reshape(-1)

        wrong = (outputs != secret_bits.to("cpu")).sum().item()
        err = 1 - wrong / secret_bits.numel()

        wrong_bch = (outputs_bch != secret_bits_bch.to("cpu")).sum().item()
        err_bch = 1 - wrong_bch / secret_bits_bch.numel()

        snr = 10 * np.log(secret_var / noise_var) / np.log(10)

        snr_list.append(snr)
        err_list.append(err)
        err_bch_list.append(err_bch)
        print(f"snr:{snr}, err:{err}, err_bch:{err_bch} ")


data_dict = {"snr": snr_list, "acc":err_list, "acc_bch":err_bch_list}
torch.save(data_dict, f"../data/err_list_2_{alpha}.pth")



# plt.plot(snr_list, err_bch_list, color=color[1], label="w/o BCH")
# plt.xlabel("SNR(dB)")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
# plt.savefig('../fig/snr.pdf', dpi=None, format='pdf')
# plt.show()
#


