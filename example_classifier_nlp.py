import argparse

from clustering_stego import ClusteringStego
from utils.test import test_classifier
from utils.train import *
from utils.init_function import *
from utils.get_data import get_sst2_data
from utils.function import get_model_params

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--model', default="Transformer", type=str, help='模型, 可选:LSTM, Transformer')
parser.add_argument('--dataset', default="sst2", type=str, help='数据集, 可选:sst2')
parser.add_argument("--alpha", default=16, type=int)
parser.add_argument("--BCH", default=True, type=bool, help='是否使用BCH编码')
parser.add_argument('--target_var', default=2e-4, type=float, help='目标方差,只在方差大于这个值的层嵌入秘密信息')
parser.add_argument('--params_num', default=2048, type=int, help='目标参数数量,只在参数数量大于这个值的层嵌入秘密信息')
args = parser.parse_args()

module = importlib.import_module("model_classifier")
# 使用 getattr 获取函数对象
cls = getattr(module, f"{args.model}")
# 加载数据集
train_loader, test_loader, max_token, vocab_len = get_sst2_data()
# 载体模型的初始化方法
init_func = init_nlp
# 初始化隐写对象
cs = ClusteringStego(init_func, target_var=args.target_var, min_nums=args.params_num, alpha=args.alpha, BCH=args.BCH)
# 载体模型
if args.model == 'Transformer':
    cov_model = cls(max_token, vocab_len)
else:
    cov_model = cls(max_token)
print(cov_model)
params1 = get_model_params(cov_model)
# 含秘初始化
secret_bits, secret_bits_bch = cs.encode(cov_model)
print(secret_bits.numel())
params2 = get_model_params(cov_model)

for i, j in zip(params1, params2):
    print(torch.var(i), torch.var(j), i.numel())

total_params = sum(p.numel() for p in cov_model.parameters())
print(f"Total number of parameters: {total_params}")

# cov_model = TransformerClassifier(max_token, vocab_len)
# cov_model = LSTM(max_token)
# torch.save(cov_model.state_dict(), f"model/{args.model}_init_original.pth")
cov_model.load_state_dict(torch.load(f"model/{args.model}_init_original.pth"))

# 加载数据集
criterion = torch.nn.CrossEntropyLoss()
optm = torch.optim.Adam(cov_model.parameters(), lr=1e-4)
# 载体模型训练
loss_list = train_classifier(cov_model, train_loader, criterion, optm, num_epochs=10)
# loss_list, extract_acc_list, extract_acc_bch_list = train_classifier_with_extract(cov_model, train_loader, criterion, optm, cs, secret_bits, secret_bits_bch, num_epochs=10)
test_classifier(cov_model, test_loader, criterion)

torch.save(cov_model.state_dict(), f"model/{args.model}_sst2_with_secret.pth")
torch.save({'secret_bits': secret_bits, 'secret_bits_bch': secret_bits_bch}, f"data/secret_{args.model}_sst2.pth")
torch.save(loss_list, f"data/loss_{args.model}_sst2_with_secret.pth")
# torch.save({'extract_acc_list': extract_acc_list, 'extract_acc_bch_list': extract_acc_bch_list}, f"data/extract_acc_{cov_model.__class__.__name__}_sst2.pth")

# 提取秘密信息
predict_secret_bits, predict_secret_bits_bch = cs.decode(cov_model)

acc = (secret_bits == predict_secret_bits).sum() / predict_secret_bits.numel()
print('原始准确率:', acc.item(), '嵌入容量:', predict_secret_bits.numel())

acc = (secret_bits_bch == predict_secret_bits_bch).sum() / predict_secret_bits_bch.numel()
print('使用bch准确率:', acc, '嵌入容量', predict_secret_bits_bch.numel())
