import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.inception import inception_v3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

inception_model = inception_v3(pretrained=False, transform_input=False).to(device)
inception_model.load_state_dict(torch.load(r'./is_fid/inception_v3.pth'))
inception_model.eval()


def inception_score(imgs: torch.Tensor, batch_size: int = 32):
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    preds, l = [], imgs.shape[0]
    for i in range(l // batch_size):
        preds += list(F.softmax(inception_model(up(imgs[i*batch_size:(i+1)*batch_size])), dim=1).cpu().detach().numpy())
    preds = torch.Tensor(preds)
    score = torch.exp(torch.sum(preds * torch.log(preds / preds.mean(axis=0)), dim=1).mean())
    return score


if __name__ == '__main__':
    imgs = torch.zeros((32, 3, 64, 64)).to(device)
    score = inception_score(imgs)
    print(score)
