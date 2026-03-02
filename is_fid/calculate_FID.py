import torch, torch.nn as nn
from torchvision.models.inception import inception_v3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

inception_model = inception_v3(pretrained=False, transform_input=False).to(device)
inception_model.load_state_dict(torch.load(r'is_fid/inception_v3.pth'))
inception_model.eval()
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)


def calculate_activation_statistics(imgs, model):
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    handle = inception_model.avgpool.register_forward_hook(hook)
    out = model(up(imgs))
    act = outputs[0].squeeze(dim=3).squeeze(dim=2)
    mu = torch.mean(act, dim=0)
    act = act - mu
    sigma = (act.transpose(0, 1) @ act) / (act.shape[1] - 1)
    handle.remove()
    return mu, sigma

def frechet_inception_distance_score(img1, img2):
    mu1_all, mu2_all, sigma1_all, sigma2_all, l = [], [], [], [], img1.shape[0]
    for i in range(l // 32):
        mu1, sigma1 = calculate_activation_statistics(img1[i*32:(i+1)*32], inception_model)
        mu2, sigma2 = calculate_activation_statistics(img2[i*32:(i+1)*32], inception_model)
        mu1_all += list(mu1.cpu().detach().numpy())
        mu2_all += list(mu2.cpu().detach().numpy())
        sigma1_all += list(sigma1.cpu().detach().numpy())
        sigma2_all += list(sigma2.cpu().detach().numpy())
    mu1_all = torch.Tensor(mu1_all)
    mu2_all = torch.Tensor(mu2_all)
    sigma1_all = torch.Tensor(sigma1_all)
    sigma2_all = torch.Tensor(sigma2_all)

    fid_score = torch.norm(mu1_all - mu2_all) + torch.trace(sigma1_all + sigma2_all - 2 * torch.sqrt(sigma1_all * sigma2_all))
    return fid_score


if __name__ == '__main__':
    img1 = torch.zeros((32, 3, 64, 64)).to(device)
    img2 = torch.ones((32, 3, 64, 64)).to(device)
    FID = frechet_inception_distance_score(img1, img2)
    print(FID)
