"""
文件名: get_data.py
作者: 徐辰屹
日期: 2024年3月6日

说明: 获取数据集的代码
"""
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk

from utils.cutout import Cutout


def get_cifar10_data(batch_size=64):
    '''
    获取 CIFAR10 数据集

    Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据

    '''
    # 定义数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，大小为32x32，填充4像素
        Cutout(6),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 下载MNIST训练集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform_train, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_mnist_data(batch_size=64):
    '''
    获取 MNIST 数据集

    Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据

    '''
    transform_mnist = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像重复三次
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 下载MNIST训练集
    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform_mnist, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform_mnist, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_mnist_data_gan(batch_size=64):
    '''
    获取 MNIST 数据集

    Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据

    '''
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    # 下载MNIST训练集
    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform_mnist, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform_mnist, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_mnist_data_vae(batch_size=64):
    '''
    获取 MNIST 数据集

    Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据

    '''
    transform_mnist = transforms.Compose([
        transforms.ToTensor()
    ])

    # 下载MNIST训练集
    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform_mnist, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform_mnist, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_fashionmnist_data(batch_size=64):
    '''
    获取 MNIST 数据集

    Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据

    '''
    transform_mnist = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像重复三次
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 下载MNIST训练集
    train_dataset = datasets.FashionMNIST(root='./dataset', train=True, transform=transform_mnist, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.FashionMNIST(root='./dataset', train=False, transform=transform_mnist, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_sst2_data(batch_size=32):
    '''
    获取 nlp 数据集

     Returns:
        train_loader: 训练集数据
        test_loader: 测试集数据
        max_token + 1: token 最大值
        vocab_len: 文本长度

    '''
    # 定义缓存目录和数据集名称
    cache_dir = 'dataset'
    dataset_name = 'sst2'
    # 统一本地数据集路径（关键：路径要和后续保存的路径一致）
    cached_dataset_path = os.path.join(cache_dir, dataset_name)

    try:
        # 尝试从本地加载（路径和cached_dataset_path统一）
        dataset = load_from_disk(cached_dataset_path)
        print(f"成功从本地加载数据集：{cached_dataset_path}")
    except FileNotFoundError:  # 精准捕获「路径不存在」异常
        print(f"本地未找到数据集，开始从网络下载：{dataset_name}")
        # 从网络下载数据集
        dataset = load_dataset(path=dataset_name)
        # 关键：将下载的数据集保存到本地，供下次加载使用
        dataset.save_to_disk(cached_dataset_path)
        print(f"数据集已下载并保存到本地：{cached_dataset_path}")
    except Exception as e:  # 捕获其他异常，方便排查
        print(f"加载数据集时发生未知错误：{str(e)}")
        raise  # 抛出异常，避免静默失败

    train_data = dataset['train']
    test_data = dataset['test']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_len = 64

    def tokenize_data(data):
        tokenized_data = tokenizer(data['sentence'], padding='max_length', truncation=True, return_tensors='pt',
                                   max_length=vocab_len)
        return tokenized_data

    train_tokenized = tokenize_data(train_data)
    test_tokenized = tokenize_data(test_data)

    train_labels = torch.tensor(train_data['label'])
    test_labels = torch.tensor(test_data['label'])

    max_token_train = train_tokenized['input_ids'].max()
    max_token_test = test_tokenized['input_ids'].max()
    max_token = max(max_token_train, max_token_test)

    train_dataset = TensorDataset(train_tokenized['input_ids'], train_labels)
    # 测试集没有标签, 所以对训练集进行划分
    test_dataset = TensorDataset(test_tokenized['input_ids'], test_labels)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # 使用random_split函数拆分数据集
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, max_token + 1, vocab_len