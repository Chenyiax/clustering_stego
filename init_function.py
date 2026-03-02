"""
文件名: init_function.py
作者: 徐辰屹
日期: 2024年5月28日

说明:
初始化函数文件
提供了各种载体模型的初始化方法
"""
import math

from torch import nn
from torch.nn import init

from utils.function import kaiming_init_

def init_alexnet(m: nn.Module, name):
    '''
    alexnet的初始化方法，详情请见: torchvision.models.alexnet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                bias_var = bound ** 2 / 3
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_densenet121(m: nn.Module, name):
    '''
    densenet的初始化方法，详情请见: torchvision.models.densenet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, _ = kaiming_init_(m.weight, a=math.sqrt(5))
        bias_var = 0
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight)
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_resnet18(m: nn.Module, name):
    '''
    resnet18的初始化方法，详情请见: torchvision.models.resnet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight, mode="fan_out", nonlinearity="relu")
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_vgg16(m: nn.Module, name):
    '''
    vgg16的初始化方法，详情请见: torchvision.models.vgg

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var = 8.2262e-05
        bias_var = 0
    elif isinstance(m, nn.Conv2d):
        weight_var, _ = kaiming_init_(m.weight, mode="fan_out", nonlinearity="relu")
        bias_var = 0
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_googlenet(m: nn.Module, name):
    return 0.0001, 0


def init_vit_b_16(m: nn.Module, name):
    '''
    visiontransformer的初始化方法，详情请见: torchvision.models.vision_transformer

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias_var = bound ** 2 / 3
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias_var = bound ** 2 / 3
    else:
        weight_var = 1
        bias_var = 0
    if 'conv_proj' in name and isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        weight_var = math.sqrt(1 / fan_in)**2
        bias_var = 0

    elif 'conv_last' in name and isinstance(m, nn.Conv2d):
        weight_var = math.sqrt(math.sqrt(2.0 / m.out_channels))
        bias_var = 0
    if "pre_logits" in name and isinstance(m, nn.Linear):
        fan_in = m.in_features
        weight_var = math.sqrt(math.sqrt(1 / fan_in))
        bias_var = 0

    if "heads" in name and isinstance(m, nn.Linear):
        weight_var = 0
        bias_var = 0

    if "mlp" in name:
        weight_var = 0.0005
        bias_var = 0.0005

    return weight_var, bias_var


def init_nlp(m: nn.Module, name):
    '''
    nlp 模型的初始化方法, 但是 nlp 没有 torchvision 这么方便的东西,所以就用了 pytorch 自带的初始化方法

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight)
    elif isinstance(m, (nn.LSTM, nn.RNN)):
        stdv = 1.0 / math.sqrt(m.hidden_size)
        weight_var = (stdv ** 2) / 3
        bias_var = weight_var
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_gan(m: nn.Module, name):
    '''
    gan网络的初始化方法

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var
