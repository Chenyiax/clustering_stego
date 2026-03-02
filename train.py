"""
文件名: train.py
作者: 徐辰屹
日期: 2024年2月1日

说明: 用于模型训练的文件
"""
import importlib

import torch
from tqdm import tqdm
from get_data import *
from classifier_model import *
from test import test_classifier
from utils.function import get_model_params
from utils.init import kaiming_init, xavier_init


def train_classifier(model, train_loader, criterion, optimizer, num_epochs=5):
    '''
    训练模型的函数
    Args:
        model: 需要被训练的模型
        train_loader: 数据集
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数

    Returns:

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_list = []
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度归零

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播

            optimizer.step()  # 更新权重

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        loss_list.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Acc:{total_correct / total_samples}")
    return loss_list

def train_classifier_with_extract(model, train_loader, criterion, optimizer, cs, secret_bits, secret_bits_bch=None, num_epochs=5):
    '''
    训练模型的函数
    Args:
        model: 需要被训练的模型
        train_loader: 数据集
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数

    Returns:

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_list = []
    extract_acc_list = []
    extract_acc_bch_list = []

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度归零

            outputs = model(inputs)  # 前向传播

            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播

            optimizer.step()  # 更新权重

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        outputs_secrets, outputs_secrets_bch = cs.decode(model)
        correct = (outputs_secrets == secret_bits).sum().item()
        accuracy = correct / outputs_secrets.numel()
        print("Extraction Accuracy of Secret Information:", accuracy)
        extract_acc_list.append(accuracy)

        if secret_bits_bch is not None:
            correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
            accuracy = correct / outputs_secrets_bch.numel()
            print("Extraction Accuracy of Secret Information BCH:", accuracy)
            extract_acc_bch_list.append(accuracy)


        epoch_loss = running_loss / len(train_loader)
        loss_list.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Acc:{total_correct / total_samples}")
    return loss_list, extract_acc_list, extract_acc_bch_list


if __name__ == '__main__':
    model_name = "resnet18"
    dataset_name = ("cifar10")
    # 动态导入模块
    module = importlib.import_module("get_data")
    # 使用 getattr 获取函数对象
    func = getattr(module, "get_" + dataset_name + "_data")
    # 加载数据集
    train_loader, test_loader = func()
    # train_loader, test_loader, vocab_size, vocab_len = get_sst2_data()


    model_class = getattr(models, model_name)
    # model = model_class(weights=None, image_size=64)
    model = model_class()
    # model = Transformer(vocab_size, vocab_len)
    model.load_state_dict(torch.load(f"model/{model_name}_init_original.pth"))
    # torch.save(model.state_dict(), f"model/{model_name}_init_original.pth")
    xavier_init(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_list = train_classifier(model, train_loader, criterion, optimizer, num_epochs=100)
    test_classifier(model, test_loader, criterion)

    torch.save(model.state_dict(), f"model/{model_name}_{dataset_name}_without_secret_xavier.pth")
    torch.save(loss_list, f"data/loss_{model_name}_{dataset_name}_without_secret_xavier.pth")

