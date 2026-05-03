import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator, FuncFormatter
from plt.mpl_config import set_style

color = set_style()
model_list = ["alexnet", "vgg16", "resnet18"]
dataset_list = ["mnist", "fashionmnist", "cifar10"]

line_width = 1.1
for dataset in dataset_list:
    for model in model_list:

        loss1 = torch.load(f"../data/loss_{model}_{dataset}_without_secret.pth")
        loss2 = torch.load(f"../data/loss_{model}_{dataset}_with_secret.pth")
        loss3 = torch.load(f"../data/loss_{model}_{dataset}_with_secret_0.pth")
        loss4 = torch.load(f"../data/loss_{model}_{dataset}_with_secret_10.pth")
        loss5 = torch.load(f"../data/loss_{model}_{dataset}_with_secret_xu.pth")[0:100]
        loss6 = torch.load(f"../data/loss_{model}_{dataset}_with_secret_hao.pth")

        plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
        plt.plot(loss1, color=color[0], label="载体模型", linewidth=line_width)

        plt.plot(loss3, color=color[5], label="Yang 等人 [18]", linewidth=line_width)
        plt.plot(loss4, color=color[4], label="Yang 等人 [19]", linewidth=line_width)
        plt.plot(loss6, color=color[3], label="Hao 等人 [21]", linewidth=line_width)
        plt.plot(loss5, color=color[2], label="第三章方法", linewidth=line_width)
        plt.plot(loss2, color=color[1], label="第四章方法", linewidth=line_width)

        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
        plt.legend()
        plt.xlabel("训练轮数")
        plt.ylabel("损失")
        plt.tight_layout()
        plt.savefig(f'../png/loss_{model}_{dataset}.png', dpi=None, format='png')
        plt.show()

for dataset in dataset_list:
    loss1 = torch.load(f"../data/loss_densenet121_{dataset}_without_secret.pth")
    loss2 = torch.load(f"../data/loss_densenet121_{dataset}_with_secret.pth")
    loss3 = torch.load(f"../data/loss_densenet121_{dataset}_with_secret_xu.pth")

    plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
    plt.plot(loss1, color=color[0], label="载体模型", linewidth=line_width)
    plt.plot(loss3, color=color[2], label="第三章方法", linewidth=line_width)
    plt.plot(loss2, color=color[1], label="第四章方法", linewidth=line_width)

    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.legend()
    plt.xlabel("训练轮数")
    plt.ylabel("损失")
    plt.tight_layout()
    plt.savefig(f'../png/loss_densenet121_{dataset}.png', dpi=None, format='png')
    plt.show()

