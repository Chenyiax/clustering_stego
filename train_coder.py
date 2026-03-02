import torch
import torch.nn as nn
import torch.optim as optim

from model import *
from utils.function import normalize_tensor
import matplotlib.pyplot as plt

num_epochs = 5000
alpha = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化自编码器和优化器
encoder = Encoder(alpha).to(device)
decoder = Decoder(alpha).to(device)

# encoder = torch.load("model/encoder.pth")
# decoder = torch.load("model/decoder.pth")

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-4)

criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()

# 训练过程
for epoch in range(num_epochs):
    params = torch.normal(0, 1, size=(2048, alpha)).to(device)
    noise = torch.normal(0, 2, size=(2048, alpha)).to(device)
    secret_bits = torch.randint(0, 2, size=(2048,)).float().to(device)

    optimizer.zero_grad()

    params_with_secret = encoder(params, secret_bits)

    params_with_secret_sort, _ = torch.sort(params_with_secret, dim=1)
    params_sort, _ = torch.sort(params, dim=1)

    params_with_noise = normalize_tensor(params_with_secret + noise)
    outputs_secrets = decoder(params_with_noise)

    # loss1 = torch.abs(params_with_secret_sort - params_sort).sum()
    # loss1 = criterion_mse(params_with_secret, params)
    loss1 = criterion_mse(params_with_secret_sort, params_sort)
    loss2 = criterion_bce(outputs_secrets, secret_bits)

    loss = loss2

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss1.item():.4f}, Loss:{loss2.item():.4f}')
        predict = (outputs_secrets > 0.5).float()
        acc = (predict == secret_bits).sum()/secret_bits.numel()
        print(acc.item())
    if epoch % 1000 == 0:
        plt.plot(params_sort[0].detach().to("cpu").numpy(), label="origin")
        plt.plot(params_with_secret_sort[0].detach().to("cpu").numpy(), label="stego")
        plt.legend()
        plt.show()

torch.save(encoder, f"model/encoder_{alpha}_without_sort.pth")
torch.save(decoder, f"model/decoder_{alpha}_without_sort.pth")

