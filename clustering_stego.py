import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.function import cut_tensor, normalize_tensor, get_secretbits_BCH, bch_decode, kaiming_init_, get_secretbits
from utils.random_generator import CustomRandomGenerator
from utils.standardizer import TensorStandardizer


class ClusteringStego:
    def __init__(self, init_function, alpha=16, batch_size=2048, target_var=2e-4, min_nums=1024, seed=42, BCH=True):
        '''
        :param init_function: 需要使用的初始化方法
        :param batch_size: 编码器生成参数时的 batch_size
        :param target_var: 最小方差提取数量
        :param min_nums: 最小参数提取数量
        '''
        self.batch_size = batch_size
        self.target_var = target_var
        self.min_nums = min_nums
        self.init_function = init_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 设置随机种子
        self.random_generator = CustomRandomGenerator(seed)
        self.alpha = alpha
        self.BCH = BCH
        self.encoder = torch.load(f"model/encoder_{self.alpha}.pth").to(self.device)
        self.decoder = torch.load(f"model/decoder_{self.alpha}.pth").to(self.device)

        # self.encoder = torch.load(f"model/encoder_{self.alpha}_without_sort.pth").to(self.device)
        # self.decoder = torch.load(f"model/decoder_{self.alpha}_without_sort.pth").to(self.device)

    def encode(self, model: torch.nn.Module):
        encoder = self.encoder
        model.to(self.device)
        secret_bits_list = []
        secret_bits_bch_list = []
        standardizer = TensorStandardizer()
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                    weight_var, bias_var = self.init_function(m, name)
                    if weight_var < self.target_var:
                        continue
                    # 如果参数过多则不生成参数
                    if m.weight.numel() < self.min_nums:
                        continue

                    # 获取秘密信息
                    if self.BCH:
                        secret_bits, secret_bits_bch = get_secretbits_BCH(m.weight.numel())
                        secret_bits_list.append(secret_bits)
                        secret_bits_bch_list.append(secret_bits_bch)
                    else:
                        secret_bits = get_secretbits(m.weight.numel(), self.alpha)
                        secret_bits_list.append(secret_bits)

                    # 参数裁剪
                    cutted_params, last_params = cut_tensor(m.weight.view(-1), secret_bits.numel() * self.alpha)
                    # 标准化
                    cutted_params = standardizer.standardize(cutted_params).view(-1, self.alpha)

                    # 创建一个 TensorDataset
                    dataset = TensorDataset(cutted_params, secret_bits)

                    # 创建一个 DataLoader 来分 batch 进行参数修改
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    params_with_secret_list = []
                    # 遍历 DataLoader
                    for batch in dataloader:
                        cutted_params_batch, secret_bits_batch = batch
                        params_with_secret = encoder(cutted_params_batch.to(self.device), secret_bits_batch)
                        params_with_secret_list.append(params_with_secret)

                    params_with_secret = torch.concatenate(params_with_secret_list)
                    params_with_secret = normalize_tensor(params_with_secret)
                    # 分布还原
                    params_with_secret = standardizer.restore(params_with_secret.view(-1))
                    params_with_secret = torch.concatenate((params_with_secret, last_params))

                    indices = self.random_generator.randperm(params_with_secret.size(0))

                    # 置乱 tensor
                    params_with_secret = params_with_secret[indices]

                    # 参数更新
                    m.weight = nn.Parameter(params_with_secret.reshape(m.weight.shape))

                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    # 统计这层的参数
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}

                    # 遍历这一层的所有参数
                    # 详情请见 pytorch 源码
                    for key, value in weight_params.items():
                        if value.numel() < self.min_nums:
                            continue
                        # 统计参数方差
                        var = torch.var(getattr(m, key)).item()
                        if var < self.target_var:
                            continue

                        # 生成秘密信息
                        if self.BCH:
                            secret_bits, secret_bits_bch = get_secretbits_BCH(value.numel())
                            secret_bits_list.append(secret_bits)
                            secret_bits_bch_list.append(secret_bits_bch)
                        else:
                            secret_bits = get_secretbits(value.numel(), self.alpha)
                            secret_bits_list.append(secret_bits)

                        weight = getattr(m, key)

                        # 参数裁剪
                        cutted_params, last_params = cut_tensor(weight.view(-1), secret_bits.numel() * self.alpha)
                        # 标准化
                        cutted_params = standardizer.standardize(cutted_params).view(-1, self.alpha)

                        # 分 batch 生成含有秘密信息的参数
                        dataset = TensorDataset(cutted_params, secret_bits)
                        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        params_with_secret_list = []
                        for batch in data_loader:
                            cutted_params_batch, secret_bits_batch = batch
                            params_with_secret = encoder(cutted_params_batch.to(self.device), secret_bits_batch)
                            params_with_secret_list.append(params_with_secret)

                        params_with_secret = torch.concatenate(params_with_secret_list)
                        params_with_secret = normalize_tensor(params_with_secret)
                        # 分布还原
                        params_with_secret = standardizer.restore(params_with_secret.view(-1))
                        params_with_secret = torch.concatenate((params_with_secret, last_params))

                        indices = self.random_generator.randperm(params_with_secret.size(0))

                        # 置乱 tensor
                        params_with_secret = params_with_secret[indices]

                        # 参数更新
                        setattr(m, key, nn.Parameter(params_with_secret.reshape(weight.shape)))

        if self.BCH:
            return torch.concatenate(secret_bits_list).view(-1), torch.concatenate(secret_bits_bch_list).view(-1)
        return torch.concatenate(secret_bits_list).view(-1), None

    def decode(self, model: torch.nn.Module):
        decoder = self.decoder
        model.to(self.device)
        secret_bits_list = []
        secret_bits_bch_list = []
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                    weight_var, bias_var = self.init_function(m, name)
                    if weight_var < self.target_var:
                        continue
                    # 如果参数过少则不生成参数
                    if m.weight.numel() < self.min_nums:
                        continue

                    params_copy = m.weight.clone().detach().view(-1)

                    indices = self.random_generator.randperm(params_copy.size(0))
                    # 还原 tensor
                    restored_tensor = params_copy[torch.argsort(indices)]

                    if self.BCH:
                        nums = m.weight.numel() // self.alpha // 128
                        cutted_params, _ = cut_tensor(restored_tensor, nums * 128 * self.alpha)
                    else:
                        nums = m.weight.numel() // self.alpha
                        cutted_params, _ = cut_tensor(restored_tensor, nums * self.alpha)
                    cutted_params = normalize_tensor(cutted_params).view(-1, self.alpha)

                    # 创建一个TensorDataset
                    dataset = TensorDataset(cutted_params)

                    # 创建一个DataLoader来处理batch
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    outputs_list = []
                    for batch in dataloader:
                        params = batch[0]
                        outputs = decoder(params.to(self.device))
                        outputs_list.append(outputs)
                    outputs = torch.concatenate(outputs_list)
                    predict = (outputs > 0.5).int().to("cpu")
                    secret_bits_list.append(predict)
                    if self.BCH:
                        predict_bch = bch_decode(predict.detach().view(-1, 128).numpy())
                        secret_bits_bch_list.append(predict_bch)

                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}
                    for key, value in weight_params.items():

                        if value.numel() < self.min_nums:
                            continue

                        params_tensor = torch.tensor([*getattr(m, key).detach().reshape(-1).tolist()],
                                                              dtype=torch.float32)

                        params_copy = params_tensor.clone().detach().view(-1)

                        indices = self.random_generator.randperm(params_copy.size(0))
                        # 还原 tensor
                        restored_tensor = params_copy[torch.argsort(indices)]

                        if self.BCH:
                            nums = value.numel() // self.alpha // 128
                            cutted_params, _ = cut_tensor(restored_tensor, nums * 128 * self.alpha)
                        else:
                            nums = value.numel() // self.alpha
                            cutted_params, _ = cut_tensor(restored_tensor, nums * self.alpha)
                        cutted_params = normalize_tensor(cutted_params).view(-1, self.alpha)

                        # 分 batch 解码
                        dataset = TensorDataset(cutted_params)
                        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        output_list = []
                        for batch in dataloader:
                            params = batch[0]
                            outputs = decoder(params.to(self.device))
                            output_list.append(outputs)
                        outputs = torch.concatenate(output_list)

                        predict = (outputs > 0.5).int().to("cpu")
                        secret_bits_list.append(predict)
                        if self.BCH:
                            predict_bch = bch_decode(predict.detach().view(-1, 128).numpy())
                            secret_bits_bch_list.append(predict_bch)

        if self.BCH:
            return torch.concatenate(secret_bits_list).view(-1), torch.concatenate(secret_bits_bch_list).view(-1)
        return torch.concatenate(secret_bits_list).view(-1), None

