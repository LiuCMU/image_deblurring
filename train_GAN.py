import os
import shutil
import tarfile
import socket
hostname = socket.gethostname()
import argparse
import wandb
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import img_dataset
from model import Conv, ResConv, Generator
from discriminator import Discriminator, Discriminator_init
from SSIM import ssim


"""The following parameters are based on:
https://wandb.ai/oilab/image_deblurring/runs/iifx4gq3?workspace=user-liu97
"""

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#parameters
if "pro" in hostname.lower():  #my mac
    train_path = "/Users/liu5/Documents/10-617HW/project/data/train"
    test_path = "/Users/liu5/Documents/10-617HW/project/data/test"
elif "exp" in hostname.lower():  #expanse 
    #copy, extract tar file to the local computing node scratch
    tar_path = "/expanse/lustre/projects/cwr109/zhen1997/img_deblur.tar"
    local_scratch = f"/scratch/{os.environ['USER']}/job_{os.environ['SLURM_JOB_ID']}"
    print("computing node local scratch path: %s" % local_scratch)
    shutil.copy(tar_path, local_scratch)
    tar = tarfile.open(os.path.join(local_scratch, "img_deblur.tar"))
    tar.extractall(local_scratch)
    tar.close()
    print("Finished extracting the dataset")
    train_path = os.path.join(local_scratch, "img_deblur/train")
    test_path = os.path.join(local_scratch, "img_deblur/test")

    device = torch.device("cuda:0")  #only using 1 GPU

elif ("braavos" in hostname.lower()) or ( "storm" in hostname.lower()):  #braavos/stromland
    train_path = "/storage/users/jack/MS_ML_datasets/img_deblur/train"
    test_path = "/storage/users/jack/MS_ML_datasets/img_deblur/test"

# train_path = "/ux1/public/wenyizha/10617project/img_deblur/train"
# test_path = "/ux1/public/wenyizha/10617project/img_deblur/test"

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
batchsize = 4



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
    netG.eval()
    with torch.no_grad():
        psnrs, ssims, sizes = [], [], []
        for batch in loader:
            xs_, ys_ = batch
            xs_ = xs_.to(device)
            ys_ = ys_.to(device)
            yhat = netG(xs_).detach()
            ys = ys_.detach()
            psnr = calc_PSNR(yhat, ys)
            ssim_i = ssim(yhat, ys).item()
            # ssim_i = 0

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
train = img_dataset(train_path, debug=False, scale=True) # debug = False in real training
train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=True) # debug = False in real training
test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))

# netG = Conv(args.num_layers, args.conv_pad, args.hidden_channels, args.pool_pad)
# netG = ResConv()
netG = Generator()
netG.to(device)
netG.apply(init_params)

netD = Discriminator()
netD.to(device)
netD.apply(init_params)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)

schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, factor=0.5, patience=patience, mode="max")
schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, factor=0.5, patience=patience, mode="max")

print("Total number of trainable parameters for Discriminator: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
print("Total number of trainable parameters for Generator: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))

results = []
G_losses = []
D_losses = []

for i in range(epochs):
    epoch = i+1
    D_learning_rate = optimizerD.param_groups[0]['lr']
    G_learning_rate = optimizerG.param_groups[0]['lr']

    netD.train()
    netG.train()

    Yhats, Ys = [], []
    # for each batch in the dataloader
    for xs, ys in train_loader:
        # update D network: maximize log(D(x))+log(1-D(G(z)))
        ## train with all-real batch
        netD.zero_grad()
        xs = xs.to(device) # real_cpu blurred image
        ys = ys.to(device) # sharp image
        N, C, H, W = ys.shape # N is b_size
        label = torch.full((N,), real_label, dtype=torch.float, device=device)
        output = netD(ys).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_sharp = output.mean().item()

        ## train with all-fake batch
        fake = netG(xs)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_deblur = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_deblur_2 = output.mean().item()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

    
    train_psnr, train_ssim = validate(train_loader)
    test_psnr, test_ssim = validate(test_loader)
    G_losses_epoch = np.mean(G_losses)
    D_losses_epoch = np.mean(D_losses)
    print("Epoch %i D LR: %.4f G LR: %.4f Training PSNR: %.2f SSIM: %.2f Test PSNR: %.2f SSIM %.2f" % (epoch, D_learning_rate, G_learning_rate, train_psnr, train_ssim, test_psnr, test_ssim))
    wandb.log({
        "learning_rate_D": D_learning_rate,
        "learning_rate_G": G_learning_rate,
        "train_psnr": train_psnr,
        "train_ssim": train_ssim,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim
    })
    results.append((epoch, D_learning_rate, G_learning_rate, train_psnr, train_ssim, test_psnr, test_ssim, G_losses_epoch, D_losses_epoch))
    if i//10 == 0:
        torch.save(netD.state_dict(), "netD_latest_paremeters.pt")
        torch.save(netG.state_dict(), "netG_latest_parameters.pt")
    schedulerD.step(test_psnr)
    schedulerG.step(test_psnr)

torch.save(netD.state_dict(), "netD_final_parameters.pt")
torch.save(netD.state_dict(), "netG_final_parameters.pt")
pickle.dump(results, open("results_GAN.pkl", "wb"))
