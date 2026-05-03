'''
文件名: kl_diver.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制 kl 散度拟合程度
'''
import argparse
import torch.nn.functional as F

import model_gan
import model_vae
from model_classifier import *
from utils import init_function
from utils.get_data import *
from utils.function import to_hist_tensor, calculate_entropy_with_hist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from plt.mpl_config import set_style

color = set_style()

line_width = 1.1
parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="densenet121", type=str,
                    help='模型, 可选:alexnet, vgg16, resnet18, densenet121, vit_b_16, LSTM, Transformer, '
                         'GeneratorGAN, GeneratorDCGAN, VAE, CVAE, Unet')
parser.add_argument('--dataset', default="cifar10", type=str, help='数据集, 可选:mnist, fashionmnist, cifar10, sst2')
parser.add_argument('--target_var', default=2e-4, type=float, help='目标方差,只比较方差大于这个值的层的参数分布')
parser.add_argument('--params_num', default=2048, type=int, help='目标参数数量,只比较参数数量大于这个值的层的参数分布')
args = parser.parse_args()

if args.model == 'vit_b_16':
    # 对于每个不同的模型,需要使用不同的初始化函数
    init_func = getattr(init_function, "init_" + args.model)
    model_class = getattr(models, args.model)
    model1 = model_class(weights=None, image_size=64)
    model2 = model_class(weights=None, image_size=64)
    model1.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_with_secret.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_without_secret.pth"))

elif args.model == 'Transformer':
    init_func = init_function.init_nlp
    vocab_size = 29668
    vocab_len = 64
    model1 = Transformer(vocab_size, vocab_len)
    model2 = Transformer(vocab_size, vocab_len)
    model1.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_with_secret.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_without_secret.pth"))
elif args.model == 'LSTM':
    init_func = init_function.init_nlp
    vocab_size = 29668
    vocab_len = 64
    model1 = LSTM(vocab_size)
    model2 = LSTM(vocab_size)
    model1.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_with_secret.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_without_secret.pth"))
elif args.model == 'GeneratorGAN' or args.model == 'GeneratorDCGAN':
    init_func = init_function.init_gan
    model1 = getattr(model_gan, args.model)(100, (1,28,28))
    model2 = getattr(model_gan, args.model)(100, (1,28,28))
    model1.load_state_dict(torch.load(f"../model/{args.model}_False.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_True.pth"))
elif args.model == 'VAE':
    init_func = init_function.init_gan
    model1 = getattr(model_vae, args.model)(28*28, 100)
    model2 = getattr(model_vae, args.model)(28*28, 100)
    model1.load_state_dict(torch.load(f"../model/{args.model}_False.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_True.pth"))
elif args.model == 'CVAE':
    init_func = init_function.init_gan
    model1 = getattr(model_vae, args.model)(100)
    model2 = getattr(model_vae, args.model)(100)
    model1.load_state_dict(torch.load(f"../model/{args.model}_False.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_True.pth"))
elif args.model == 'Unet':
    init_func = init_function.init_gan
    model1 = torch.load(f"../model/{args.model}_False.pth")
    model2 = torch.load(f"../model/{args.model}_True.pth")

else:
    # 对于每个不同的模型,需要使用不同的初始化函数
    init_func = getattr(init_function, "init_" + args.model)
    model_class = getattr(models, args.model)
    model1 = model_class(weights=None)
    model2 = model_class(weights=None)
    model1.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_with_secret.pth"))
    model2.load_state_dict(torch.load(f"../model/{args.model}_{args.dataset}_without_secret.pth"))


print(model1)
def collect_params_with_criteria(model, params_list, args, init_func):
    """收集满足条件的模型参数"""
    with torch.no_grad():  # 禁用梯度计算
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                weight_var, bias_var = init_func(module, name)

                # 参数过少或者方差过小则跳过
                if module.weight.numel() < args.params_num or weight_var < args.target_var:
                    continue

                params_list.append(module.weight.detach().reshape(-1).to("cpu"))
            if isinstance(module, (nn.LSTM, nn.RNN)):
                temp_list = []
                weight_params = {name: param for name, param in module.named_parameters() if 'weight' in name}
                for key, value in weight_params.items():
                    if value.numel() < args.params_num:
                        continue
                    # 统计参数方差
                    var = torch.var(getattr(module, key)).item()
                    if var < args.target_var:
                        continue
                    temp_list.append(getattr(module, key).detach().reshape(-1).to("cpu"))
                params_list.append(torch.concatenate(temp_list))


# 收集带有秘密信息和不带秘密信息的参数
params_with_secret = []
params_without_secret = []
collect_params_with_criteria(model1, params_with_secret, args, init_func)
collect_params_with_criteria(model2, params_without_secret, args, init_func)

num_params = len(params_with_secret)
k = 0
kl_list = []
entropy_list = []
for i, j in zip(params_without_secret, params_with_secret):
    bins = int(math.sqrt(len(i)))
    hist_tensor1, bin_centers1 = to_hist_tensor(i, bins)
    hist_tensor2, bin_centers2 = to_hist_tensor(j, bins)

    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
    kl_list.append(kl_divergence)
    entropy1 = calculate_entropy_with_hist(hist_tensor1)
    entropy2 = calculate_entropy_with_hist(hist_tensor2)
    entropy_list.append(torch.abs(entropy1 - entropy2))
    # print(kl_divergence)

    plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
    plt.plot(bin_centers1, hist_tensor1, color=color[1], label='载体模型', linewidth=line_width)
    plt.plot(bin_centers2, hist_tensor2, color=color[0], label='隐写模型', linewidth=line_width)
    plt.xlabel('参数值')
    plt.ylabel('频率')
    plt.title(f"KL = {kl_divergence:.4f}")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))  # 自动选择范围(nbins表示有几个刻度)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../png/{args.model}_{args.dataset}_kl_{k}.png', dpi=None, format='png')
    plt.close()
    k += 1

kl_list = torch.tensor(kl_list)
entropy_list = torch.tensor(entropy_list)

print(torch.mean(entropy_list))
print(torch.mean(kl_list))


