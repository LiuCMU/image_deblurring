import socket
hostname = socket.gethostname()
import argparse
import wandb
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import img_dataset
from model import ResConv
from SSIM import ssim


"""The following parameters are based on:
https://wandb.ai/oilab/image_deblurring/runs/iifx4gq3?workspace=user-liu97
"""

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#parameters
if "pro" in hostname.lower():  #my mac
    train_path = "/Users/liu5/Documents/10-617HW/project/data/train"
    test_path = "/Users/liu5/Documents/10-617HW/project/data/test"
elif "exp" in hostname.lower():  #expanse 
    train_path = "/expanse/lustre/projects/cwr109/zhen1997/data/train"
    test_path = "/expanse/lustre/projects/cwr109/zhen1997/data/test"
elif "braavos" in hostname.lower():  #braavos
    train_path = "/storage/users/jack/MS_ML_datasets/img_deblur/train"
    test_path = "/storage/users/jack/MS_ML_datasets/img_deblur/test"


parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--conv_pad", type=int, default=1)
parser.add_argument("--hidden_channels", type=int, default=20)
parser.add_argument("--pool_pad", type=int, default=2)
args = parser.parse_args()
wandb.init(project="img_deblur", entity="liu97", config=args)
config = wandb.config

lr = 0.0005
patience = 10
epochs = 100
batchsize = 16
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def calc_PSNR(img1, img2):
    """
    calculating PSNR based on 10.3390/s20133724
    img1/img1: tensor of shape (N, C, H, W)
    """
    N, C, H, W = img1.shape
    denominator = (1/(N*C*H*W)) * torch.sum(torch.square(img1 - img2))
    psnr = 10*torch.log10((255*255)/denominator)
    return psnr.item()


def validate(loader):
    model.eval()
    with torch.no_grad():
        psnrs, ssims, sizes = [], [], []
        for batch in loader:
            xs_, ys_ = batch
            xs_ = xs_.to(device)
            ys_ = ys_.to(device)
            yhat = model(xs_).detach()
            ys = ys_.detach()
            psnr = calc_PSNR(yhat, ys)
            ssim_i = ssim(yhat, ys).item()

            psnrs.append(psnr)
            ssims.append(ssim_i)
            sizes.append(len(batch))
    score = np.dot(psnrs, sizes)/np.sum(sizes)
    ssim_score = np.dot(ssims, sizes)/np.sum(sizes)
    return (score, ssim_score)


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

#load datasets
train = img_dataset(train_path, debug=False, scale=True)
train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=True)
test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))

# model = Conv(config.num_layers, config.conv_pad, config.hidden_channels, config.pool_pad)
model = ResConv()
model.to(device)
model.apply(init_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode="max")
print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

results = []
for i in range(epochs):
    epoch = i+1
    learning_rate = optimizer.param_groups[0]['lr']
    

    model.train()
    Yhats, Ys = [], []
    for xs, ys in train_loader:
        xs = xs.to(device)
        ys = ys.to(device)
        yhat = model(xs)
        N, C, H, W = ys.shape
        denominator = (1/(N*C*H*W)) * torch.sum(torch.square(ys - yhat))
        loss = -10*torch.log10((255*255)/denominator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    train_psnr, train_ssim = validate(train_loader)
    test_psnr, test_ssim = validate(test_loader)
    print("Epoch %i LR: %.4f Training PSNR: %.2f SSIM: %.2f Test PSNR: %.2f SSIM %.2f" % (epoch, learning_rate,  train_psnr, train_ssim, test_psnr, test_ssim))
    wandb.log({
        "learning_rate": learning_rate,
        "train_psnr": train_psnr,
        "train_ssim": train_ssim,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim
    })
    results.append((epoch, learning_rate,  train_psnr, train_ssim, test_psnr, test_ssim))
    if i//10 == 0:
        torch.save(model.state_dict(), "latest_res.pt")
    scheduler.step(test_psnr)
torch.save(model.state_dict(), "model_parameters_res.pt")
pickle.dump(results, open("results_res.pkl", "wb"))
