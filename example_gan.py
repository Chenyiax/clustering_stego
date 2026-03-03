import argparse
import os

from torch import Tensor
from torchvision.utils import save_image

from tqdm import tqdm

from clustering_stego import ClusteringStego
from model_gan import *
from utils.get_data import get_mnist_data_gan
from utils.init_function import init_gan
from is_fid import inception_score, frechet_inception_distance_score


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
parser.add_argument("--lr", type=float, default=5e-5, help="优化器学习率")
parser.add_argument("--latent_dim", type=int, default=100, help="输入噪声维度")
parser.add_argument("--img_size", type=int, default=28, help="图像大小")
parser.add_argument("--channels", type=int, default=1, help="图像通道数")
parser.add_argument("--embedding_secret", type=bool, default=True, help="是否嵌入秘密信息")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数
adversarial_loss = torch.nn.BCELoss()

# 初始化生成器和解码器
generator = GeneratorDCGAN(opt.latent_dim, img_shape)
discriminator = DiscriminatorDCGAN(img_shape)

# torch.save(generator.state_dict(), f"model/{generator.__class__.__name__}_init_original.pth")
generator.load_state_dict(torch.load(f"model/{generator.__class__.__name__}_init_original.pth"))

total_params = sum(p.numel() for p in generator.parameters())
print(f"Total number of parameters: {total_params}")

if opt.embedding_secret:
    # 载体模型的初始化方法
    init_func = init_gan
    # 初始化隐写对象
    cs = ClusteringStego(init_func, target_var=5e-4)
    # 嵌入秘密信息
    secret_bits, secret_bits_bch = cs.encode(generator)
    print(secret_bits.numel(), secret_bits_bch.numel())


generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

dataloader, _ = get_mnist_data_gan(opt.batch_size)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))


# ----------
#  训练
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in tqdm(enumerate(dataloader), leave=False):
        # 生成标签
        valid = Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake = Tensor(imgs.size(0), 1).fill_(0.0).to(device)
        # 训练集图像为真实图像
        real_imgs = imgs.to(device)

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()

        # 输入噪声
        z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)

        # 生成一个 batch 的图像
        gen_imgs = generator(z)

        # 生成器的损失为判别器判定为假的损失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练 gan 网络的判别器
        # ---------------------

        optimizer_D.zero_grad()

        # 判别器损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, d_loss.item(), g_loss.item())
    )

save_image(gen_imgs.data[:25], f"images/{generator.__class__.__name__}_{opt.embedding_secret}.png", nrow=5, normalize=True)


# 假设opt和generator已经定义
z = torch.randn(64, opt.latent_dim).to("cuda")

# 生成图像
gen_imgs = generator(z).expand(-1, 3, -1, -1)

# 计算Inception Score
is_score = inception_score(gen_imgs)

# 从dataloader中获取一个batch的数据
data, labels = next(iter(dataloader))
data = data.expand(-1, 3, -1, -1).float().to("cuda")
# 计算Frechet Inception Distance Score
fid = frechet_inception_distance_score(data, gen_imgs)

print("FID:", fid, "IS:", is_score)

if opt.embedding_secret:
    # 提取秘密信息
    predict_secret_bits, predict_secret_bits_bch = cs.decode(generator)
    acc = (secret_bits == predict_secret_bits).sum() / predict_secret_bits.numel()
    print('原始准确率:', acc.item(), '嵌入容量:', predict_secret_bits.numel())

    acc = (secret_bits_bch == predict_secret_bits_bch).sum() / predict_secret_bits_bch.numel()
    print('使用bch准确率:', acc, '嵌入容量', predict_secret_bits_bch.numel())

torch.save(generator.state_dict(), f'model/{generator.__class__.__name__}_{opt.embedding_secret}.pth')
# torch.save(discriminator, f'model/{discriminator.__class__.__name__}_{opt.embedding_secret}.pth')