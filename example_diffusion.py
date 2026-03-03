import argparse

import torch
from torchvision.utils import save_image

from clustering_stego import ClusteringStego
from utils.get_data import get_mnist_data_vae
from utils.init_function import init_gan
from is_fid import inception_score, frechet_inception_distance_score
from utils.denoising_diffusion_pytorch import Unet, GaussianDiffusion

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
parser.add_argument("--lr", type=float, default=5e-5, help="优化器学习率")
parser.add_argument("--img_size", type=int, default=28, help="图像大小")
parser.add_argument("--channels", type=int, default=1, help="图像通道数")
parser.add_argument("--embedding_secret", type=bool, default=False, help="是否嵌入秘密信息")
opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim=64,
    channels=1,
    dim_mults=(1, 2, 2)
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

diffusion = GaussianDiffusion(
    model,
    objective='pred_noise',
    image_size=28,
    timesteps=500
)
diffusion.to(device)

# torch.save(model.state_dict(), f"model/{model.__class__.__name__}_init_original.pth")
model.load_state_dict(torch.load(f"model/{model.__class__.__name__}_init_original.pth"))

if opt.embedding_secret:
    # 载体模型的初始化方法
    init_func = init_gan
    # 初始化隐写对象
    cs = ClusteringStego(init_func)
    # 嵌入秘密信息
    secret_bits, secret_bits_bch = cs.encode(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
train_loader,_ = get_mnist_data_vae()
epochs = opt.n_epochs
for epoch in range(epochs):
    print('Epoch: ', epoch + 1)
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        loss = diffusion(images)

        if step % 100 == 0:
            print("Step: ", step + 1, "Loss:", loss.item())

        loss.backward()
        optimizer.step()

# torch.save(model, f'model/{model.__class__.__name__}_{opt.embedding_secret}.pth')

generated_images = diffusion.sample(batch_size=64)
gen_imgs = generated_images.reshape(-1, 1, 28, 28)

# save_image(gen_imgs.data[:25], f"images/diffusion_{opt.embedding_secret}.png", nrow=5, normalize=True)

gen_imgs = gen_imgs.expand(-1, 3, -1, -1).float()
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
