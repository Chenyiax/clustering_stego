import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils import prune
from torchvision import models
from tqdm import tqdm

from get_data import *
from init_function import init_resnet18
from clustering_stego import ClusteringStego

init_func = init_resnet18
cs = ClusteringStego(init_func, target_var=2e-4)
secrets = torch.load("data/secret_ResNet18_cifar10.pth")
secret_bits = secrets["secret_bits"]
secret_bits_bch = secrets["secret_bits_bch"]
acc_list = []
acc_list_bch = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None).to(device)
model.load_state_dict(torch.load("model/ResNet18_cifar10_with_secret.pth"))
train_loader, _ = get_mnist_data()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

outputs_secrets, outputs_secrets_bch = cs.decode(model)
correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
acc_list.append(accuracy)
correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy_bch = correct / outputs_secrets_bch.numel()
acc_list_bch.append(accuracy_bch)
print(accuracy, accuracy_bch)

for i in range(0, 99):
    model.to(device)
    for inputs, labels in tqdm(train_loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 梯度归零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

    outputs_secrets, outputs_secrets_bch = cs.decode(model)

    correct = (outputs_secrets == secret_bits).sum().item()
    accuracy = correct / outputs_secrets.numel()
    acc_list.append(accuracy)

    correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
    accuracy_bch = correct / outputs_secrets_bch.numel()
    acc_list_bch.append(accuracy_bch)
    print(accuracy, accuracy_bch)

torch.save(acc_list, "data/acc_fine-tuning_cross_5e-5.pth")
torch.save(acc_list_bch, "data/acc_bch_fine-tuning_cross_5e-5.pth")
