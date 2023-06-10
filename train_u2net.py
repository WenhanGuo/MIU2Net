#! -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

from model_u2net import u2net_full
import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np
# import cupy
# from cupyx.scipy.ndimage import gaussian_filter as gaus_blur
from scipy.ndimage import gaussian_filter as gaus_blur
import pandas as pd
from glob import glob1
import astropy.io.fits as fits

# define training device (cpu/gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

# Dataset class for our own data structure
class ImageDataset(Dataset):
    def __init__(self, catalog, data_dir, transform=None, target_transform=None):
        """
        catalog: name of .csv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transform: transformations to input data (gamma) prior to training
        target_transform: transformation to target data (kappa) prior to training 
        """
        self.img_names = pd.read_csv(catalog)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_names)
    
    def read_img(self, idx, img_type=['gamma1', 'gamma2', 'kappa']):
        img_name = self.img_names[img_type][idx]   # get single image name
        img_path = os.path.join(self.data_dir, img_type, img_name)
        with fits.open(img_path, memmap=False) as f:
            img = f[0].data   # read image into numpy array
        return np.float32(img)   # force apply float32 to resolve endian conflict
    
    def __getitem__(self, idx):
        # read in images
        gamma1 = self.read_img(idx, img_type='gamma1')
        gamma2 = self.read_img(idx, img_type='gamma2')
        kappa = self.read_img(idx, img_type='kappa')
        # reformat data shapes
        gamma = np.array([gamma1, gamma2])
        gamma = np.moveaxis(gamma, 0, 2)
        kappa = np.expand_dims(kappa, 2)
        # apply transforms
        if self.transform:
            gamma = self.transform(gamma)
        if self.target_transform:
            kappa = self.target_transform(kappa)
        # gamma shape: torch.Size([2, 512, 512]); kappa shape: torch.Size([1, 512, 512])
        return gamma, kappa


# add gaussian noise to shear
class AddGaussianNoise(object):
    def __init__(self, n_galaxy, mean=0.):
        """
        calculate the gaussian noise standard deviation to be added to shear.
        please refer to https://articles.adsabs.harvard.edu/pdf/2004MNRAS.350..893H Eq.12. 面积替换为方形的
        noise_std^2 = {sigma_e^2 / 2} / {θ_G^2 * n_galaxy}
        """
        self.n_galaxy = n_galaxy
        self.mean = mean

    def __call__(self, tensor):
        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * self.n_galaxy)
        std = np.sqrt(variance)
        return tensor + torch.randn(tensor.size()) * std + self.mean
        # for 50 galaxies per pix, std = 0.1951; 
        # for 20 galaxies per pix, std = 0.3085


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子, 
        注意在训练开始之前, pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


# set hyperparameters
def set_parameters():
    global epoch, batch_size, learning_rate, gamma, weight_decay, delta
    global data_dir

    epoch = 64
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-5
    delta = 0.01

    data_dir = '/share/lirui/Wenhan/WL/data_new'
    return


if __name__ == '__main__':
    # set hyperparameters
    set_parameters()

    # prepare train and validation datasets
    ds_args = dict(data_dir=data_dir, 
                   transform=Compose([ToTensor(), 
                                      AddGaussianNoise(n_galaxy=50)
                                      ]), 
                   target_transform=Compose([ToTensor()]))
    train_data = ImageDataset(catalog=os.path.join(data_dir, 'train.csv'), **ds_args)
    val_data = ImageDataset(catalog=os.path.join(data_dir, 'validation.csv'), **ds_args)
    # prepare train and validation dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=True, drop_last=True, **loader_args)

    # initialize UNet model
    model = u2net_full()
    model = model.to(memory_format=torch.channels_last)
    # data parallel training on multiple GPUs (restrained by cuda visible devices)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=device)
    torch.cuda.empty_cache()

    # setting loss function, optimizer, and scheduler
    loss_fn = nn.HuberLoss(delta=delta)
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, 
                                  weight_decay=weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epoch,
                                       warmup=True, warmup_epochs=3)
    scaler = torch.cuda.amp.GradScaler()   # Use torch.cuda.amp for mixed precision training

    # use tensorboard to visualize computation
    writer = SummaryWriter("logs_train")
    # delete existing tensorboard logs
    old_logs = glob1('./logs_train', '*')
    for f in old_logs:
        os.remove(os.path.join(os.getcwd(), 'logs_train', f))

    # begin training
    total_train_step = 0
    best_loss = False
    for i in range(epoch):
        print(f"--------------------------Starting epoch {i+1}--------------------------")
        val_loss = 0.0

        # training steps
        model.train()
        for gamma, kappa in train_loader:
            gamma = gamma.to(device, memory_format=torch.channels_last)
            # gamma shape: (batchsize, 2, 512, 512)
            kappa = kappa.to(device, memory_format=torch.channels_last)
            # kappa shape: (batchsize, 1, 512, 512)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(gamma)
                loss_targ = loss_fn(outputs[0], kappa)
                losses = [loss_targ]

                # original_imgs = cupy.asarray(kappa)
                # outputs_cupy = cupy.asarray(outputs)
                original_imgs = kappa.cpu()
                def append_gb_loss(device, outputs, targets, losses_list, side_id, sigma):
                    blurred_imgs = gaus_blur(targets, sigma=sigma, order=0)
                    blurred_imgs = torch.tensor(blurred_imgs)
                    blurred_imgs = blurred_imgs.to(device, memory_format=torch.channels_last)

                    loss_side = loss_fn(outputs[side_id], blurred_imgs)
                    losses_list.append(loss_side)

                gb_args = dict(device=device, outputs=outputs, targets=original_imgs, losses_list=losses)
                append_gb_loss(side_id=1, sigma=1, **gb_args)
                append_gb_loss(side_id=2, sigma=2, **gb_args)
                append_gb_loss(side_id=3, sigma=6, **gb_args)
                append_gb_loss(side_id=4, sigma=12, **gb_args)
                append_gb_loss(side_id=5, sigma=20, **gb_args)
                append_gb_loss(side_id=6, sigma=50, **gb_args)
                
                train_step_loss = sum(losses)
                
                print('targ_loss =', losses[0].item())
                print('loss 1 =', losses[1].item())
                print('loss 2 =', losses[2].item())
                print('loss 3 =', losses[3].item())
                print('loss 4 =', losses[4].item())
                print('loss 5 =', losses[5].item())
                print('loss 6 =', losses[6].item())
                print('total loss =', train_step_loss.item())

                # losses = [loss_fn(outputs[i], kappa) for i in range(len(outputs))]
                # side_loss = sum(losses) - losses[0]   # total loss for all side maps
                # targ_loss = losses[0]   # target prediction loss
                # train_step_loss = sum(losses)
                # print('total, side, targ =', train_step_loss, side_loss, targ_loss)

            # optimizer step
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(train_step_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                train_step_loss.backward()
                optimizer.step()
            lr_scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

            total_train_step += 1
            if total_train_step % 50 == 0:
                print(f"train step: {total_train_step}, \
                      current loss: {train_step_loss.item()}")
        
        # validation steps
        model.eval()
        with torch.no_grad():
            for gamma, kappa in val_loader:
                gamma = gamma.to(device, memory_format=torch.channels_last)
                kappa = kappa.to(device, memory_format=torch.channels_last)
                outputs = model(gamma)
                val_step_loss = loss_fn(outputs, kappa)
                val_loss += val_step_loss.item()

        # printing epoch stats & writing to tensorboard
        print(f"epoch training loss = {train_step_loss}", f"LR = {curr_lr}")
        print(f"avg validation loss = {val_loss/len(val_loader)}")
        writer.add_scalar("train_loss", train_step_loss, global_step=i+1)
        writer.add_scalar("val_loss", val_loss/len(val_loader), global_step=i+1)
        writer.add_scalar("lr", curr_lr, global_step=i+1)

        # save model for every best loss epoch
        if not best_loss:
            best_loss = val_loss
        elif val_loss < best_loss:
            torch.save(model, f'./models/best_epoch{i+1}.pth')
            print(f"saved best loss model at epoch = {i+1}!")
            best_loss = val_loss
            best_epoch = i+1
    
    print(f"best epoch number is {best_epoch}.")
    writer.close()
