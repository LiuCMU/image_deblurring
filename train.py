import socket
hostname = socket.gethostname()
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import img_dataset
from model import Conv


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#parameters
if "pro" in hostname.lower():  #my mac
    train_path = "/Users/liu5/Documents/10-617HW/project/data/train"
    test_path = "/Users/liu5/Documents/10-617HW/project/data/test"
elif "exp" in hostname.lower():  #expanse 
    train_path = "/expanse/lustre/projects/cwr109/zhen1997/data/train"
    test_path = "/expanse/lustre/projects/cwr109/zhen1997/data/test"

lr = 0.0005
patience = 10
epochs = 100
batchsize = 128
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
    denominator = (1/(N*C*H*W)) * np.sum(np.square(img1 - img2))
    psnr = 10*np.log10((255*255)/denominator)
    return psnr


def validate(loader):
    model.eval()
    with torch.no_grad():
        psnrs, sizes = [], []
        for batch in loader:
            xs_, ys_ = batch
            xs_ = xs_.to(device)
            ys_ = ys_.to(device)
            yhat = model(xs_).detach().to("cpu").numpy()
            ys = ys_.detach().to("cpu").numpy()
            psnr = calc_PSNR(yhat, ys)

            psnrs.append(psnr)
            sizes.append(len(batch))
    score = np.dot(psnrs, sizes)/np.sum(sizes)
    return score


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

#load datasets
train = img_dataset(train_path, debug=False, scale=True)
train_loader = DataLoader(train, batchsize)
test = img_dataset(test_path, debug=False, scale=True)
test_loader = DataLoader(test, batchsize)

model = Conv()
model.to(device)
model.apply(init_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode="min")
print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

nspr = 999
results = []
for i in range(epochs):
    epoch = i+1
    learning_rate = optimizer.param_groups[0]['lr']
    scheduler.step(nspr)

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

    train_psnr = validate(train_loader)
    test_psnr = validate(test_loader)
    print("Epoch %i LR: %.4f Training NSPR: %.2f Test NSPR: %.2f" % (epoch, learning_rate,  train_psnr, test_psnr))
    results.append((epoch, learning_rate,  train_psnr, test_psnr))
    torch.save(model.state_dict(), "latest.pt")
torch.save(model.state_dict(), "model_parameters.pt")
pickle.dump(results, open("results.pkl", "wb"))
