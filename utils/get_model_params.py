"""
文件名: generate_params.py
作者: 徐辰屹
日期: 2024年3月18日

说明:
生成含秘密信息的模型参数或不含秘密信息的模型参数
"""
import argparse
import torch
from torchvision import models
from clustering_stego import ClusteringStego
from utils.get_data import get_cifar10_data

from utils.test import test_classifier
from utils.train import train_classifier
from utils.function import get_model_params
from utils.init_function import init_resnet18

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--with_secret', default=False, type=bool, help='生成的参数是否含有秘密信息')
parser.add_argument('--lr', default=1e-4, type=float, help='载体模型的学习率')
parser.add_argument('--epoch', default=100, type=int, help='载体模型在目标数据集上的训练轮数')

args = parser.parse_args()

train_loader, test_loader = get_cifar10_data()
task_model = models.resnet18().to("cuda")
init_func = init_resnet18
cs = ClusteringStego(init_func)

params = get_model_params(task_model)

# 判断哪些层嵌入了秘密信息
position = []
i = 0
for param in params:
    if param.numel() > cs.max_nums or param.numel() < cs.min_nums or torch.var(param) < cs.target_var:
        position.append(i)
    i += 1

if args.with_secret:
    # 生成并嵌入秘密信息
    secret_bits = cs.encode(task_model)

# 载体模型训练
print("training task model:")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(task_model.parameters(), lr=args.lr)
train_classifier(task_model, train_loader, criterion, optimizer, num_epochs=args.epoch, with_secret=args.with_secret)
test_classifier(task_model, test_loader, criterion)
task_model.to("cpu")

# 提取秘密信息的参数
params = get_model_params(task_model)

# 删除没有嵌入秘密信息的层
for index in sorted(position, reverse=True):
    del params[index]

if args.with_secret:
    torch.save(params, 'data/params_with_secret.pth')
else:
    torch.save(params, 'data/params_without_secret.pth')