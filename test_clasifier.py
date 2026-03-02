import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder
from utils.function import normalize_tensor
import matplotlib.pyplot as plt

num_epochs = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化自编码器和优化器
encoder = torch.load("model/encoder.pth").to(device)
decoder = torch.load("model/decoder.pth").to(device)

params = torch.normal(0, 1, size=(1024, 8)).to(device)
secret_bits = torch.randint(0, 2, size=(1024, 1)).float().to(device)
position = torch.linspace(0, 1, steps=8).repeat(1024, 1).to(device)

position_with_secret = encoder(params, secret_bits)
_, indices = torch.sort(position_with_secret)

params_with_secret = torch.gather(params, 1, indices)
features = torch.concatenate((params_with_secret, position), dim=1)
outputs = decoder(features)
predict = (outputs > 0.5).float()
acc = (secret_bits == predict).sum()/secret_bits.numel()
print(acc)