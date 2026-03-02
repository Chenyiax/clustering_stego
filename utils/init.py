from torch.nn import init
import math


def kaiming_init(model, a=0, mode='fan_in', nonlinearity='relu'):
    """
    对PyTorch模型进行Kaiming初始化

    参数:
        model (nn.Module): 需要进行初始化的PyTorch模型
        a (float): 激活函数的负斜率，对于ReLU，a=0
        mode (str): 可选'fan_in'或'fan_out'，指定计算增益时使用的扇入/扇出方式
        nonlinearity (str): 非线性激活函数名称，默认为'relu'
    """

    def init_func(m):
        classname = m.__class__.__name__
        # 对卷积层进行Kaiming初始化
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=a, mode=mode, nonlinearity=nonlinearity)
            # 如果有偏置项，将偏置初始化为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 对线性层进行Kaiming初始化
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=a, mode=mode, nonlinearity=nonlinearity)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 对批归一化层的权重初始化为1，偏置初始化为0
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    # 应用初始化函数到模型的所有子模块
    model.apply(init_func)
    return model





def xavier_init(model, gain=1.0, distribution='uniform'):
    """
            对PyTorch模型进行Xavier初始化（Glorot初始化）

            参数:
                model (nn.Module): 需要初始化的PyTorch模型
                gain (float): 增益系数，用于调整初始化范围，默认1.0
                distribution (str): 初始化分布，可选'uniform'（均匀分布）或'normal'（正态分布）
            """

    def init_func(m):
        classname = m.__class__.__name__

        # 处理卷积层
        if classname.find('Conv') != -1:
            # 计算卷积层的扇入（输入维度）和扇出（输出维度）
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]

            if distribution == 'uniform':
                # 均匀分布初始化：[-limit, limit]，其中limit = gain * sqrt(6 / (fan_in + fan_out))
                limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
                init.uniform_(m.weight.data, -limit, limit)
            elif distribution == 'normal':
                # 正态分布初始化：均值0，方差 = gain^2 / (fan_in + fan_out)
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                init.normal_(m.weight.data, mean=0.0, std=std)
            else:
                raise ValueError(f"不支持的分布类型: {distribution}，请使用'uniform'或'normal'")

            # 偏置项初始化为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # 处理线性层（全连接层）
        elif classname.find('Linear') != -1:
            # 计算线性层的扇入和扇出
            fan_in = m.in_features
            fan_out = m.out_features

            if distribution == 'uniform':
                limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
                init.uniform_(m.weight.data, -limit, limit)
            elif distribution == 'normal':
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                init.normal_(m.weight.data, mean=0.0, std=std)

            # 偏置项初始化为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # 批归一化层沿用标准初始化（权重1，偏置0）
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    # 应用初始化到模型的所有子模块
    model.apply(init_func)
    return model
