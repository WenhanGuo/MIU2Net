#! -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

from model_u2net import u2net_full
import torch
from torch import nn
import transforms as T
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader
from my_dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

import math
from glob import glob1
from prettytable import PrettyTable

# define training device (cpu/gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


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

    epoch = 128
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-4
    delta = 0.01

    data_dir = '/share/lirui/Wenhan/WL/data_new'
    return


if __name__ == '__main__':
    # set hyperparameters
    set_parameters()

    # prepare train and validation datasets; augment data with flip & rotations; add noise
    train_args = dict(data_dir=data_dir, 
                      transforms=T.Compose([
                          T.ToTensor(), 
                          T.RandomHorizontalFlip(prob=0.5), 
                          T.RandomVerticalFlip(prob=0.5), 
                          T.DiscreteRotation(angles=[0,90,180,270])
                          ]), 
                      gaus_noise=T.AddGaussianNoise(n_galaxy=50)
                      )
    valid_args = dict(data_dir=data_dir, 
                      transforms=T.Compose([
                          T.ToTensor()
                          ]), 
                      gaus_noise=T.AddGaussianNoise(n_galaxy=50)
                      )
    train_data = ImageDataset(catalog=os.path.join(data_dir, 'train.csv'), **train_args)
    val_data = ImageDataset(catalog=os.path.join(data_dir, 'validation.csv'), **valid_args)
    # prepare train and validation dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=True, drop_last=True, **loader_args)

    # initialize UNet model
    model = u2net_full()
    model = model.to(memory_format=torch.channels_last)
    # count_parameters(model)
    # data parallel training on multiple GPUs (restrained by cuda visible devices)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=device)
    torch.cuda.empty_cache()

    # setting loss function, optimizer, and scheduler
    loss_fn = nn.HuberLoss(delta=delta)
    loss_fn = loss_fn.to(device)
    blur1 = GaussianBlur(kernel_size=5, sigma=(2.0, 3.0))
    blur2 = GaussianBlur(kernel_size=11, sigma=(2.0, 3.0))
    blur3 = GaussianBlur(kernel_size=21, sigma=(4.0, 6.0))
    blur4 = GaussianBlur(kernel_size=41, sigma=(6.0, 8.0))
    blur5 = GaussianBlur(kernel_size=91, sigma=(12.0, 16.0))
    blur6 = GaussianBlur(kernel_size=151, sigma=(25.0, 30.0))

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
    loss_list = [[],[],[],[],[],[],[]]
    
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
                losses.append(loss_fn(outputs[1], blur1(kappa)))
                losses.append(loss_fn(outputs[2], blur2(kappa)))
                losses.append(loss_fn(outputs[3], blur3(kappa)))
                losses.append(loss_fn(outputs[4], blur4(kappa)))
                losses.append(loss_fn(outputs[5], blur5(kappa)))
                losses.append(loss_fn(outputs[6], blur6(kappa)))

                train_step_loss = sum(losses)
                
                loss_list[0].append(losses[0].item())
                # print('targ_loss =', losses[0].item())
                # print(f'{losses[1].item()},{losses[2].item()},{losses[3].item()},{losses[4].item()},{losses[5].item()},{losses[6].item()}')
                # print('total loss =', train_step_loss.item())

                loss_list[1].append(losses[1].item())
                loss_list[2].append(losses[2].item())
                loss_list[3].append(losses[3].item())
                loss_list[4].append(losses[4].item())
                loss_list[5].append(losses[5].item())
                loss_list[6].append(losses[6].item())

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
                      total loss: {train_step_loss.item():.3}, \
                      targ loss: {losses[0].item():.3},\n \
                      side loss: {losses[1].item():.3},{losses[2].item():.3},{losses[3].item():.3},{losses[4].item():.3},{losses[5].item():.3},{losses[6].item():.3}")
        
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
