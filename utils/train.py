"""
文件名: train.py
作者: 徐辰屹
日期: 2024年2月1日

说明: 用于模型训练的文件
"""
import importlib

from tqdm import tqdm
from model_classifier import *


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