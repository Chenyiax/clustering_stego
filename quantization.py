from collections import OrderedDict
from copy import deepcopy

import torch
from torchvision.models import resnet18

from clustering_stego import ClusteringStego
from get_data import get_cifar10_data
from init_function import init_resnet18
from utils.function import get_model_params
from utils.random_generator import CustomRandomGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
# 创建一个预训练模型
model = torch.load("./model/ResNet18_cifar10_with_secret.pth")

init_func = init_resnet18
cs = ClusteringStego(init_func, target_var=2e-4)
secrets = torch.load("data/secret_ResNet18_cifar10.pth")
secret_bits = secrets["secret_bits"]
secret_bits_bch = secrets["secret_bits_bch"]

model_params1 = get_model_params(model)

def signed_quantize(x, bits, bias=None):
    min_val, max_val = x.min(), x.max()
    n = 2.0 ** (bits -1)
    scale = max(abs(min_val), abs(max_val)) / n
    qx = torch.floor(x / scale)
    if bias is not None:
        qb = torch.floor(bias / scale)
        return qx, qb
    else:
        return qx

# 对模型整体进行量化
def scale_quant_model(model, bits):
    net = deepcopy(model)
    params_quant = OrderedDict()
    params_save = OrderedDict()

    for k, v in model.state_dict().items():
        if 'classifier' not in k and 'num_batches' not in k and 'running' not in k:
            if 'weight' in k:
                weight = v
                bias_name = k.replace('weight', 'bias')
                try:
                    bias = model.state_dict()[bias_name]
                    w, b = signed_quantize(weight, bits, bias)
                    params_quant[k] = w
                    params_quant[bias_name] = b
                    if bits > 8 and bits <= 16:
                        params_save[k] = w.short()
                        params_save[bias_name] = b.short()
                    elif bits > 1 and bits <= 8:
                        params_save[k] = w.char()
                        params_save[bias_name] = b.char()
                    elif bits == 1:
                        params_save[k] = w.bool()
                        params_save[bias_name] = b.bool()
                    print(1)

                except:
                    # Ensure 'w' is assigned
                    w = signed_quantize(weight, bits)
                    params_quant[k] = w
                    params_save[k] = w.char()

        else:
            params_quant[k] = v
            params_save[k] = v
    net.load_state_dict(params_quant)
    return net, params_save

model, params_save = scale_quant_model(model, 1)

model_params2 = get_model_params(model)

outputs_secrets, outputs_secrets_bch = cs.decode(model)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()

correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy_bch = correct / outputs_secrets_bch.numel()

print(accuracy, accuracy_bch)
