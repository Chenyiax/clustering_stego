import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import models

from utils.init_function import init_resnet18
from plt.mpl_config import set_style
color = set_style()

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="resnet18", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121, vit_b_16, LSTM, Transformer')
parser.add_argument('--dataset', default="cifar10", type=str, help='数据集, 可选:mnist, fashionmnist, cifar10, sst2')
parser.add_argument('--target_var', default=2e-4, type=float, help='目标方差,只比较方差大于这个值的层的参数分布')
parser.add_argument('--params_num', default=2048, type=int, help='目标参数数量,只比较参数数量大于这个值的层的参数分布')
args = parser.parse_args()

model1 = models.resnet18()
model1.load_state_dict(torch.load("../model/resnet18_cifar10_with_secret.pth"))

model2 = models.resnet18()
model2.load_state_dict(torch.load("../model/resnet18_cifar10_without_secret.pth"))
print(model1)
init_func = init_resnet18
params1 = model1.conv1.weight.detach().numpy()
params2 = model2.conv1.weight.detach().numpy()

X = params1.reshape(192, 49)
Y = params2.reshape(192, 49)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Y_pca = pca.fit_transform(Y_scaled)


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='#1f77b4', alpha=0.7, marker='o', label="隐写卷积核")
plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c='#ff7f0e', alpha=0.7, marker='x', label="载体卷积核")
plt.legend()
plt.savefig("../png/pca.png", format="png")
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))
