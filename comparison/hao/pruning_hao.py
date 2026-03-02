import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
from torchvision.models import resnet18, vgg16, alexnet
from tqdm import tqdm
from torchvision import models
from get_data import get_mnist_data
from utils.cutout import Cutout

# 设置设备 (CUDA or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument("--capacity", default=384, type=int)
parser.add_argument("--beta", default=500000, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--dataset", default="cifar10", type=str)
args = parser.parse_args()

# 密钥矩阵
def generate_random_sparse_matrix(rows, cols, sparsity=0.2, value_range=(0.00, 0.15)):
    """
    生成一个随机稀疏矩阵。

    参数:
        rows (int): 矩阵行数。
        cols (int): 矩阵列数。
        sparsity (float): 稀疏率，非零元素所占比例 (0 到 1)。
        value_range (tuple): 非零元素值的范围 (min, max)。

    返回:
        torch.sparse.FloatTensor: 生成的稀疏矩阵。
    """
    # 计算非零元素的数量
    num_nonzeros = int(rows * cols * sparsity)

    # 随机生成非零元素的位置
    indices = torch.vstack((
        torch.randint(0, rows, (num_nonzeros,)),
        torch.randint(0, cols, (num_nonzeros,))
    ))

    # 随机生成非零元素的值
    values = torch.rand(num_nonzeros) * (value_range[1] - value_range[0]) + value_range[0]

    # 构造稀疏矩阵
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols))
    return sparse_matrix

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim)).to(self.device)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim)).to(self.device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, mean=0, std=1)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, input_dim=25, output_dim=5):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, adjacency, conv):
        kernel_size_height = conv.weight.shape[2]
        kernel_size_width = conv.weight.shape[3]
        feature = torch.reshape(conv.weight, (-1, kernel_size_height * kernel_size_width))

        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        logits = self.sig(logits)
        return logits

# 定义嵌入与提取模块
class StegoModule:
    def __init__(self, capacity, m: nn.Module):
        self.capacity = capacity
        self.secret_bits = torch.tensor(np.rint(np.random.rand(1, capacity)), dtype=torch.float32)
        self.kernel_size = m.weight.shape[2] * m.weight.shape[3]
        self.conv_features = m.weight.shape[0] * m.weight.shape[1]
        if capacity/self.conv_features != int(capacity/self.conv_features):
            print("capacity is not divisible by conv_features")
            exit(0)
        self.gcn_net = GcnNet(input_dim=self.kernel_size, output_dim=int(capacity/self.conv_features))
        self.adjacency_matrix = generate_random_sparse_matrix(self.conv_features, self.conv_features).to(device)

    def generate_watermark(self):
        return self.secret_bits

    def extract_watermark(self, m: nn.Module):
        decoder_data = self.gcn_net(self.adjacency_matrix, m)
        return decoder_data.view(-1)

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


module = importlib.import_module("get_data")
# 使用 getattr 获取函数对象
func = getattr(module, "get_"+args.dataset+"_data")
# 加载数据集
train_loader, test_loader = func()

# 创建载体网络模型
model_class = getattr(models, args.model)
model = model_class(weights=None)
model.load_state_dict(torch.load(f"model/{args.model}_{args.dataset}_with_secret_hao.pth"))

model.to(device)
print(model)


if args.model == 'resnet18':
    stegolayer = model.conv1
elif args.model == 'vgg16':
    stegolayer = model.features[0]
elif args.model =='alexnet':
    stegolayer = model.features[0]
elif args.model == 'densenet121':
    stegolayer = model.features[5].conv
else:
    stegolayer = None

# 隐写模块
stego_module = StegoModule(args.capacity, stegolayer)

data = torch.load(f"data/secret_{args.model}_{args.dataset}_hao.pth")
stego_module.gcn_net = torch.load("../../model/gcn_resnet18_cifar10.pth")
stego_module.secret_bits = data['secret_bits']
stego_module.adjacency_matrix = data['matrix']

predict = stego_module.extract_watermark(stegolayer)
predictions = (predict > 0.5).float()
secrets = stego_module.generate_watermark().to(device)
correct = (secrets == predictions).sum().item()
accuracy = correct / secrets.numel()
print(accuracy)

num_epochs = 100

acc_list = []
for epoch in range(num_epochs):
    prune_ratio = 0.01 * epoch

    weight = stegolayer.weight
    total_weights = weight.size(0) * weight.size(1) * weight.size(2) * weight.size(3)
    num_prune = int(total_weights * prune_ratio)
    flat_weight = weight.flatten().to("cpu").detach().numpy()

    # 基于权重绝对值的大小排序
    sorted_indices = np.argsort(np.abs(flat_weight))
    prune_indices = sorted_indices[:num_prune]

    # 随机选择要剪枝的索引
    # prune_indices = np.random.choice(total_weights, num_prune, replace=False)

    flat_weight[prune_indices] = 0
    weight = torch.tensor(flat_weight).view(weight.shape).to(device)
    stegolayer.weight = nn.Parameter(weight)

    extract_watermark = stego_module.extract_watermark(stegolayer)


    predictions = (extract_watermark > 0.5).float()
    correct = (secrets == predictions).sum().item()
    accuracy = correct / secrets.numel()
    acc_list.append(accuracy)

    print("Epoch:", epoch, "Extraction Accuracy of Secret Information:", accuracy)


torch.save(acc_list, f"data/less_pruning_acc_{args.model}_hao.pth")