import argparse
import importlib
import time


from clustering_stego import ClusteringStego
from train import *
import init_function
from utils.function import get_model_params
from torchvision import models

from utils.init import kaiming_init, xavier_init

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="densenet121", type=str, help='模型, 可选:alexnet, vgg16, resnet18, densenet121, vit_b_16')
parser.add_argument('--dataset', default="cifar10", type=str, help='数据集, 可选:mnist, fashionmnist, cifar10')
parser.add_argument("--alpha", default=16, type=int)
parser.add_argument("--BCH", default=True, type=bool, help='是否使用BCH编码')
parser.add_argument("--lr", default=1e-4, type=float, help='隐写模型学习率')
parser.add_argument('--target_var', default=2e-4, type=float, help='目标方差,只在方差大于这个值的层嵌入秘密信息')
parser.add_argument('--params_num', default=2048, type=int, help='目标参数数量,只在参数数量大于这个值的层嵌入秘密信息')
args = parser.parse_args()

# 动态导入模块
module = importlib.import_module("get_data")
# 使用 getattr 获取函数对象
func = getattr(module, "get_"+args.dataset+"_data")
# 加载数据集
train_loader, test_loader = func()
# 不同的数据集,训练轮数不同
epoch_dict = {'mnist': 10, 'fashionmnist': 30, 'cifar10': 100}

# 载体模型的初始化方法
init_func = getattr(init_function, "init_"+args.model)

# 初始化隐写对象
cs = ClusteringStego(init_func, target_var=args.target_var, min_nums=args.params_num, BCH=args.BCH, alpha=args.alpha)

# 载体模型
model_class = getattr(models, args.model)

if args.model == 'vit_b_16':
    model = model_class(weights=None, image_size=64)
else:
    model = model_class(weights=None)

print(model)

model.load_state_dict(torch.load(f"model/{args.model}_init_original.pth"))
# torch.save(model.state_dict(), f"model/{args.model}_init_original.pth")

# xavier_init(model)

# 含秘初始化
start_time = time.time()
secret_bits, secret_bits_bch = cs.encode(model)
end_time = time.time()
print(end_time - start_time)
print("Total number of secrets:", secret_bits.numel())

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 加载数据集
criterion = torch.nn.CrossEntropyLoss()
optm = torch.optim.Adam(model.parameters(), lr=args.lr)
# 载体模型训练
loss_list = train_classifier(model, train_loader, criterion, optm, num_epochs=epoch_dict[args.dataset])
# loss_list, extract_acc_list, extract_acc_bch_list = train_classifier_with_extract(model, train_loader, criterion, optm, cs, secret_bits, secret_bits_bch, num_epochs=100)
test_classifier(model, test_loader, criterion)

torch.save(model.state_dict(), f"model/{args.model}_{args.dataset}_with_secret_var{args.target_var}.pth")
torch.save({'secret_bits': secret_bits, 'secret_bits_bch': secret_bits_bch}, f"data/secret_{args.model}_{args.dataset}_var{args.target_var}.pth")
torch.save(loss_list, f"data/loss_{args.model}_{args.dataset}_with_secret_var{args.target_var}.pth")
# torch.save({'extract_acc_list': extract_acc_list, 'extract_acc_bch_list': extract_acc_bch_list}, f"data/extract_acc_{args.model}_{args.dataset}_var{args.target_var}.pth")

# 提取秘密信息
predict_secret_bits, predict_secret_bits_bch = cs.decode(model)

acc = (secret_bits == predict_secret_bits).sum() / predict_secret_bits.numel()
print('原始准确率:', acc.item(), '嵌入容量:', predict_secret_bits.numel())

acc = (secret_bits_bch == predict_secret_bits_bch).sum() / predict_secret_bits_bch.numel()
print('使用bch准确率:', acc, '嵌入容量', predict_secret_bits_bch.numel())
