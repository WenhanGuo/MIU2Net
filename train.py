#! -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

from model import UNet
import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
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
        return gamma, kappa


# add gaussian noise to shear
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.0424):
        """
        mean and standard deviation for the gaussian noise to be added to shear
        for 50 galaxies per square arcmin, std = 0.3/sqrt(50) = 0.0424
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# set hyperparameters
def set_parameters():
    global epoch, batch_size, learning_rate, gamma, weight_decay, delta
    global data_dir

    epoch = 60
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-3
    delta = 0.01

    data_dir = '/share/lirui/Wenhan/WL/data_new'
    return


if __name__ == '__main__':
    # set hyperparameters
    set_parameters()

    # prepare train and validation datasets
    ds_args = dict(data_dir=data_dir, 
                   transform=Compose([ToTensor(), 
                                    #   AddGaussianNoise(mean=0, std=0.0424)
                                    #   AddGaussianNoise(mean=0, std=0.0671)
                                      AddGaussianNoise(mean=0, std=0.0949)
                                      ]), 
                   target_transform=Compose([ToTensor()]))
    train_data = ImageDataset(catalog=os.path.join(data_dir, 'train.csv'), **ds_args)
    val_data = ImageDataset(catalog=os.path.join(data_dir, 'validation.csv'), **ds_args)
    # prepare train and validation dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=True, drop_last=True, **loader_args)

    # initialize UNet model
    model = UNet()
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
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # use tensorboard to visualize computation
    writer = SummaryWriter("logs_train")
    # delete existing tensorboard logs
    old_logs = glob1('./logs_train', '*')
    for f in old_logs:
        os.remove(os.path.join(os.getcwd(), 'logs_train', f))

    # begin training
    total_train_step = 0
    for i in range(epoch):
        print(f"--------------------------Starting epoch {i+1}--------------------------")
        train_loss, val_loss = 0.0, 0.0

        # training steps
        model.train()
        for gamma, kappa in train_loader:
            gamma = gamma.to(device, memory_format=torch.channels_last)
            kappa = kappa.to(device, memory_format=torch.channels_last)
            outputs = model(gamma)
            train_step_loss = loss_fn(outputs, kappa)

            # optimizer step
            optimizer.zero_grad()
            train_step_loss.backward()
            optimizer.step()
            train_loss += train_step_loss.item()

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
        # curr_lr = optimizer.param_groups[0]['lr']

        # printing epoch stats & writing to tensorboard
        print(f"epoch training loss = {train_loss/len(train_loader)}")
        print(f"avg validation loss = {val_loss/len(val_loader)}")
        writer.add_scalar("train_loss", train_loss/len(train_loader), global_step=i+1)
        writer.add_scalar("val_loss", val_loss/len(val_loader), global_step=i+1)

        torch.save(model, f'./models/wenhan_epoch{i+1}.pth')

    writer.close()
