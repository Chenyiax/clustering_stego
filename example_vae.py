import argparse

import torch.optim as optim
from torchvision.utils import save_image

from clustering_stego import ClusteringStego
from utils.get_data import get_mnist_data_vae
from utils.init_function import init_gan
from is_fid import inception_score, frechet_inception_distance_score
from model_vae import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VAE", help="模型名称, 可选VAE, CVAE")
parser.add_argument("--n_epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
parser.add_argument("--lr", type=float, default=5e-5, help="优化器学习率")
parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
parser.add_argument("--img_size", type=int, default=28, help="图像大小")
parser.add_argument("--channels", type=int, default=1, help="图像通道数")
parser.add_argument("--embedding_secret", type=bool, default=True, help="是否嵌入秘密信息")
opt = parser.parse_args()
print(opt)

train_loader,_ = get_mnist_data_vae(opt.batch_size)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = opt.img_size ** 2

if opt.model_name == "VAE":
    model = VAE(input_dim, opt.latent_dim)
elif opt.model_name == "CVAE":
    model = CVAE(opt.latent_dim)
else:
    raise ValueError("model_name must be VAE or CVAE")

# torch.save(model.state_dict(), f"model/{model.__class__.__name__}_init_original.pth")
model.load_state_dict(torch.load(f"model/{model.__class__.__name__}_init_original.pth"))

if opt.embedding_secret:
    # 载体模型的初始化方法
    init_func = init_gan
    # 初始化隐写对象
    cs = ClusteringStego(init_func, target_var=5e-4)
    # 嵌入秘密信息
    secret_bits, secret_bits_bch = cs.encode(model)

model.to(device)
criterion = nn.MSELoss(reduction='sum')               # 重构损失使用二元交叉熵
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 训练模型
def loss_function(recon_x, x, mu, logvar):
    x = x.view(-1, input_dim)
    recon_x = recon_x.view(-1, input_dim)
    # 计算重构损失（二元交叉熵）
    BCE = criterion(recon_x, x)
    # 计算 KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



for epoch in range(opt.n_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{opt.n_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}")

# 使用训练好的模型生成数据
model.eval()
with torch.no_grad():
    # 从潜在空间采样并解码生成图像
    z = torch.randn(25, opt.latent_dim).to(device)
    generated_images = model.decode(z).view(-1, 1, 28, 28)
    save_image(generated_images, f"png/{model.__class__.__name__}_{opt.embedding_secret}.png", nrow=5,normalize=True)

# 假设opt和generator已经定义
z = torch.randn(64, opt.latent_dim).to("cuda")

# 生成图像
gen_imgs = model.decode(z).view(-1, 1, 28, 28).expand(-1, 3, -1, -1)

# 计算Inception Score
is_score = inception_score(gen_imgs)
# 从dataloader中获取一个batch的数据
data, labels = next(iter(train_loader))
data = data.expand(-1, 3, -1, -1).float().to("cuda")
# 计算Frechet Inception Distance Score
fid = frechet_inception_distance_score(data, gen_imgs)

print("FID:", fid, "IS:", is_score)


if opt.embedding_secret:
    # 提取秘密信息
    predict_secret_bits, predict_secret_bits_bch = cs.decode(model)
    acc = (secret_bits == predict_secret_bits).sum() / predict_secret_bits.numel()
    print('原始准确率:', acc.item(), '嵌入容量:', predict_secret_bits.numel())

    acc = (secret_bits_bch == predict_secret_bits_bch).sum() / predict_secret_bits_bch.numel()
    print('使用bch准确率:', acc, '嵌入容量', predict_secret_bits_bch.numel())

torch.save(model.state_dict(), f'model/{model.__class__.__name__}_{opt.embedding_secret}.pth')