import math
import bchlib
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain


def normalize_tensor(tensor):
    # 保存原始形状
    original_shape = tensor.shape

    # 将张量展平
    flat_tensor = tensor.view(-1)

    # 计算均值和标准差
    mean = flat_tensor.mean()
    std = flat_tensor.std()

    # 标准化
    normalized_flat_tensor = (flat_tensor - mean) / std

    # 将展平后的张量还原成原来的形状
    normalized_tensor = normalized_flat_tensor.view(original_shape)

    return normalized_tensor


def cut_tensor(tensor: torch.Tensor, target):
    '''
    将 tensor 裁剪成两份

    :param tensor: 目标 tensor
    :param target: 目标数
    :return: 裁剪完成的 tensor
    '''
    trimmed_tensor = tensor[:target]
    last = tensor[target:]
    return trimmed_tensor, last


def get_model_params(model: torch.nn.Module) -> list:
    '''
    获取神经网络参数的函数

    Args:
        model(nn.Module) : 目标神经网络模型

    Returns:
        list: 一个列表,长度为符合条件的层的参数， 其中每一个元素为一个tensor
        例: 一个10层的神经网络, 返回值为一个长度为 10 的 list, 每一个元素都是一个tensor, 为这一层的参数
    '''
    last_params_list = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # 获取更新后的模型参数
            if m.bias is None:
                last_params_list.append(m.weight.detach().reshape(-1).to("cpu"))
            else:
                last_params_list.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))
    return last_params_list


def to_hist_tensor(tensor: torch.Tensor, bins: int) -> (torch.Tensor, np.ndarray):
    '''
    将张量转换为直方图的函数
    Args:
        tensor(torch.Tensor) :输入张量
        bins(int) :直方图个数

    Returns:
        tensor: 张量的直方图
        ndarray: 对应直方图所代表的横坐标, 用于绘图
    '''
    hist, bin_edges = np.histogram(tensor.detach().to("cpu").numpy(), bins=bins, range=(-0.5, 0.5))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    prob_dist = hist / hist.sum()
    prob_dist[prob_dist == 0] = 1e-6
    prob_dist = torch.tensor(prob_dist, dtype=torch.float32)
    return prob_dist, bin_centers


def kaiming_init_(
    tensor: torch.Tensor, a: float = math.sqrt(3), mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):
    '''
    凯明初始化方差计算函数
    详情请见 torch.nn.init.kaiming_uniform_

    Args:
        tensor(torch.Tensor): 待初始化的张量
        a(float): leaky_relu的斜率，如果nonlinearity是relu的话，这项参数没用
        mode(str): fan_in是优化前向传播, fan_out是优化反向传播
        nonlinearity(str): 激活函数用的是relu还是leaky_relu

    Returns:
        float: 这层权重所需要服从的方差
        float: 对应的偏置所需要服从的方差(如果有的话)
    '''
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias_var = bound**2/3
    return std**2, bias_var

def bytearray_to_int_list(byte_array):
    '''
    字节转比特流的函数

    Args:
        byte_array: 字节数组

    Returns:
        list: 只含有 0 和 1 的list
    '''
    int_list = []
    for byte in byte_array:
        # 将每个字节拆分为8位，并将每位转换为整数
        for i in range(7, -1, -1):
            int_list.append((byte >> i) & 1)
    return int_list


def bch_encode(data: np.ndarray) -> np.ndarray:
    '''
    bch 编码函数
    bchlib这个库真的很阴间, 注释写太少了

    Args:
        data(np.ndarray): 只含有 0 和 1 的np数组, 长度应当为 64

    Returns:
        np.ndarray: 经过bch编码后的np数组, 只含有0和1, 长度为 128
    '''
    outputs = []
    bch = bchlib.BCH(10, m=7)
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        ecc = bytearray(bch.encode(byte_stream))
        encoded_data = bytearray_to_int_list(byte_stream + ecc)
        outputs.append(encoded_data)
    del bch
    return np.array(outputs)


def bch_decode(data: np.ndarray) -> torch.Tensor:
    '''
    bch 解码函数
    bchlib这个库真的很阴间, 注释写太少了

    Args:
        data(np.ndarray): 只含有 0 和 1 的np数组, 长度应当为 128

    Returns:
        np.ndarray: 经过bch纠错后的np数组, 只含有0和1, 长度为 64
    '''
    outputs = []
    bch = bchlib.BCH(10, m=7)
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        data, ecc = byte_stream[:-bch.ecc_bytes], byte_stream[-bch.ecc_bytes:]
        nerr = bch.decode(data, ecc)
        bch.correct(data, ecc)
        decoded_data = bytearray_to_int_list(data)
        outputs.append(decoded_data)
    del bch
    return torch.tensor(outputs)

# 生成秘密信息的函数
def get_secretbits_BCH(nums: int) -> (torch.Tensor, torch.Tensor):
    secret_nums = nums // 16
    batch = secret_nums // 128
    random_binary = np.random.randint(0, 2, size=(batch, 56))
    binary = bch_encode(random_binary)
    return torch.tensor(binary, dtype=torch.float32).view(-1), torch.tensor(random_binary, dtype=torch.float32).view(-1)

def get_secretbits(nums: int, alpha=16) -> (torch.Tensor, torch.Tensor):
    secret_nums = nums // alpha
    random_binary = np.random.randint(0, 2, size=(secret_nums))
    return torch.tensor(random_binary, dtype=torch.float32).view(-1)


def calculate_entropy_with_hist(tensor: torch.Tensor, bins: int = None) -> torch.Tensor:
    '''
    基于to_hist_tensor函数计算输入张量的熵值
    计算逻辑参考文档IV-B2节：熵值衡量参数分布不确定性，公式为H(X) = -ΣP(x_i)·log2(P(x_i))
    Args:
        tensor(torch.Tensor): 输入参数张量（如文档中模型层的初始化参数），形状不限（内部会展平处理）
        bins(int, optional): 直方图区间个数，默认采用文档"平方根法则"（基于参数数量计算），确保分布估计可靠

    Returns:
        torch.Tensor: 熵值（单位：比特），值越接近表明两组参数的分布随机特性越一致（符合文档评估逻辑）
    '''
    # 步骤1：展平输入张量，统一处理维度（适配模型层参数的任意形状，如卷积核参数、全连接层参数）
    tensor_flat = tensor.flatten()
    num_params = tensor_flat.shape[0]

    # 步骤2：确定直方图区间个数（默认遵循文档IV-B2节"平方根法则"，与KL散度计算的bin size逻辑一致）
    if bins is None:
        bins = int(np.sqrt(num_params))
        # 避免参数数量过少导致bin数不足（文档实验中排除<2048参数的层，此处确保bin数≥10以保证估计可靠性）
        bins = max(bins, 10)

    # 步骤3：调用to_hist_tensor获取参数的概率分布（归一化直方图）
    prob_dist, _ = to_hist_tensor(tensor_flat, bins=bins)

    # 步骤4：计算熵值（文档定义的比特熵，使用log2；通过clamp避免数值异常）
    log_prob = torch.log2(prob_dist.clamp(min=1e-12))  # 限制最小概率，防止log2(极小值)导致数值溢出
    entropy = -torch.sum(prob_dist * log_prob)

    return entropy

def calculate_kl_divergence(p: torch.Tensor, q: torch.Tensor, bins: int = None) -> float:
    '''
    基于to_hist_tensor函数计算输入张量的KL散度值
    计算逻辑参考文档IV-B2节：KL散度衡量参数分布差异，公式为D_KL(P||Q) = ΣP(x_i)·log2(P(x_i)/Q(x_i))
    Args:
        p(torch.Tensor): 输入参数张量（如文档中模型层的初始化参数），形状不限（内部会展平处理）
        q(torch.Tensor): 输入参数张量（如文档中模型层的置乱参数），形状不限（内部会展平处理）
        bins(int, optional): 直方图区间个数，默认采用文档"平方根法则"（基于参数数量计算），确保分布估计可靠

    Returns:
        float: KL散度值（单位：比特），值越大表明两组参数的分布差异越大（符合文档评估逻辑）
    '''
    # 步骤1：展平输入张量，统一处理维度（适配模型层参数的任意形状，如卷积核参数、全连接层参数）
    p_flat = p.flatten()
    q_flat = q.flatten()
    num_params = p_flat.shape[0]

    # 步骤2：确定直方图区间个数（默认遵循文档IV-B2节"平方根法则"，与KL散度计算的bin size逻辑一致）
    if bins is None:
        bins = int(np.sqrt(num_params))
        # 避免参数数量过少导致bin数不足（文档实验中排除<2048参数的层，此处确保bin数≥10以保证估计可靠性）
        bins = max(bins, 10)

    # 步骤3：调用to_hist_tensor获取参数的概率分布（归一化直方图）
    p_prob_dist, _ = to_hist_tensor(p_flat, bins=bins)
    q_prob_dist, _ = to_hist_tensor(q_flat, bins=bins)


    kl_divergence = F.kl_div(p_prob_dist.log(), q_prob_dist, reduction='sum')
    return kl_divergence.item()
