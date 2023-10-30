#! -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 4'

from model_u2net import u2net_full
import torch
from torch import nn
import transforms as T
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader
from my_dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

import shutil
import math
import numpy as np
from prettytable import PrettyTable
import datetime


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
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


def generate_peak_mask(args, target, thres_std=1):
        thresholds = torch.mean(target, dim=(1,2,3)) + thres_std * torch.std(target, dim=(1,2,3))
        thresholds = thresholds.view(args.batch_size, 1, 1, 1)
        target_mask = target > thresholds   # mask shape: (batchsize, 1, 512, 512)
        u2net_sides_mask = target_mask.repeat(7, 1, 1, 1, 1)
        return target_mask, u2net_sides_mask


def main(args):

    # prepare train and validation datasets; augment data with flip & rotations; add noise
    if args.gaus_blur == True:
        shear_gb = GaussianBlur(kernel_size=5, sigma=2.0)
    else:
        shear_gb = None

    train_data = ImageDataset(catalog=os.path.join(args.dir, 'train.ecsv'), 
                              n_galaxy=args.n_galaxy, 
                              transforms=T.Compose([
                                  T.KS_rec(args), 
                                  T.RandomHorizontalFlip(prob=0.5), 
                                  T.RandomVerticalFlip(prob=0.5), 
                                  T.ContinuousRotation(degrees=180), 
                                  T.RandomCrop(size=512), 
                                  T.Wiener(args), 
                                  T.sparse(args), 
                                  T.MCALens(args)
                                  ]), 
                              gaus_blur=shear_gb
                              )
    val_data = ImageDataset(catalog=os.path.join(args.dir, 'validation.ecsv'), 
                            n_galaxy=args.n_galaxy, 
                            transforms=T.Compose([
                                T.KS_rec(args), 
                                T.RandomCrop(size=512), 
                                T.Wiener(args), 
                                T.sparse(args), 
                                T.MCALens(args)
                                ]), 
                            gaus_blur=shear_gb
                            )
    # prepare train and validation dataloaders
    loader_args = dict(batch_size=args.batch_size, num_workers=args.cpu, pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=True, drop_last=True, **loader_args)

    # initialize UNet model
    in_channels = 2
    if args.ks == 'add':
        in_channels += 1
    if args.wiener == 'add':
        in_channels += 1
    if args.sparse == 'add':
        in_channels += 1
    if args.mcalens == 'add':
        in_channels += 1
    print('in_channels =', in_channels)
    model = u2net_full(in_ch=in_channels)

    if args.param_count == True:
        count_parameters(model)
    # data parallel training on multiple GPUs (restrained by cuda visible devices)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device, memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    # setting loss function
    if args.loss_fn == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'Huber':
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
    loss_fn = loss_fn.to(device)

    # setting optimizer & lr scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, 
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=3)
    scaler = torch.cuda.amp.GradScaler()   # Use torch.cuda.amp for mixed precision training

    # use tensorboard to visualize computation
    writer = SummaryWriter('../tlogs_kappa2d')
    # delete existing tensorboard logs
    shutil.rmtree('../tlogs_kappa2d')
    os.mkdir('../tlogs_kappa2d')

    # begin training
    total_train_step = 0
    best_loss = False
    for i in range(args.epochs):
        print(f"--------------------------Starting epoch {i+1}--------------------------")
        # training steps
        model.train()
        for gamma, kappa in train_loader:
            gamma = gamma.to(device, memory_format=torch.channels_last)
            # gamma shape: (batchsize, 2, 512, 512)
            kappa = kappa.to(device, memory_format=torch.channels_last)
            # kappa shape: (batchsize, 1, 512, 512)
            kappa_mask, outputs_mask = generate_peak_mask(args, target=kappa, thres_std=1)
            kappa_peak = kappa * kappa_mask

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(gamma)
                outputs_peak = [outputs[n] * outputs_mask[n] for n in range(len(outputs))]

                loss_targ = loss_fn(outputs[0], kappa)
                loss_targ_peak = loss_fn(outputs_peak[0], kappa_peak)
                # native mode: based on true kappa for every side output
                if args.loss_mode == 'native':
                    losses = [loss_fn(outputs[j], kappa) for j in range(len(outputs))]
                    losses.insert(0, loss_targ)
                    losses_peak = [loss_fn(outputs_peak[k], kappa_peak) for k in range(len(outputs))]
                    losses_peak.insert(0, loss_targ_peak)
                
                train_step_loss = sum(losses) + sum(losses_peak) * 5

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
            # print 3 train losses per epoch
            if total_train_step % (len(train_loader) // 3) == 0:
                print(f"train step: {total_train_step}, \
                      total loss: {train_step_loss:.3}, \
                      peak loss: {sum(losses_peak):.3},\n \
                      targ loss: {losses[0]:.3}, \
                      targ peak loss: {losses_peak[0]:.3},\n \
                      side loss: {losses[1]:.3},{losses[2]:.3},{losses[3]:.3},{losses[4]:.3},{losses[5]:.3},{losses[6]:.3},\n \
                      side peak loss: {losses_peak[1]:.3},{losses_peak[2]:.3},{losses_peak[3]:.3},{losses_peak[4]:.3},{losses_peak[5]:.3},{losses_peak[6]:.3}")
        
        # validation steps
        model.eval()
        val_loss, val_peak_loss = 0.0, 0.0
        with torch.no_grad():
            for gamma, kappa in val_loader:
                gamma = gamma.to(device, memory_format=torch.channels_last)
                kappa = kappa.to(device, memory_format=torch.channels_last)
                kappa_mask, _ = generate_peak_mask(args, target=kappa, thres_std=1)
                kappa_peak = kappa * kappa_mask
                outputs = model(gamma)
                outputs_peak = outputs * kappa_mask
                val_loss += loss_fn(outputs, kappa)
                val_peak_loss += loss_fn(outputs_peak, kappa_peak)
                val_step_loss = val_loss + val_peak_loss * 5

        # printing final 1x1 convolution layer (learned weights for each side outputs)
        for name, param in model.named_parameters():
            # if single gpu, name == 'out_conv.weight'; if multiple gpus, name == 'module.out_conv.weight'
            if param.requires_grad and 'out_conv.weight' in name:
                last_w = np.array(torch.flatten(param.data.cpu()))
            elif param.requires_grad and 'out_conv.bias' in name:
                last_b = param.data.item()
                print(f'1x1 conv weights = {last_w.round(3)}, bias = {last_b:.3}')

        # printing epoch stats & writing to tensorboard
        print(f"epoch training loss = {train_step_loss:.3}, base {sum(losses):.3}, peak {sum(losses_peak):.3}", f"LR = {curr_lr:.3}")
        print(f"avg validation loss = {val_step_loss/len(val_loader):.4}, base {val_loss/len(val_loader):.4}, peak {val_peak_loss/len(val_loader):.4}")
        writer.add_scalars("train_loss", {'total_step_loss':train_step_loss.item(), 
                                          'base_loss':sum(losses).item(), 
                                          'peak_loss':sum(losses_peak).item()}, global_step=i+1)
        writer.add_scalars("targ_loss", {'base_loss':losses[0].item(), 
                                         'peak_loss':losses_peak[0].item()}, global_step=i+1)
        writer.add_scalars("side_losses", {'side1':losses[1].item(), 
                                           'side2':losses[2].item(), 
                                           'side3':losses[3].item(), 
                                           'side4':losses[4].item(), 
                                           'side5':losses[5].item(), 
                                           'side6':losses[6].item()}, global_step=i+1)
        writer.add_scalars("side_peak_losses", {'side1':losses_peak[1].item(), 
                                                'side2':losses_peak[2].item(), 
                                                'side3':losses_peak[3].item(), 
                                                'side4':losses_peak[4].item(), 
                                                'side5':losses_peak[5].item(), 
                                                'side6':losses_peak[6].item()}, global_step=i+1)
        writer.add_scalars("conv_weights", {'side1':last_w[0], 
                                            'side2':last_w[1],
                                            'side3':last_w[2],
                                            'side4':last_w[3],
                                            'side5':last_w[4],
                                            'side6':last_w[5]}, global_step=i+1)
        writer.add_scalars("val_loss", {'total_step_loss':(val_step_loss/len(val_loader)).item(), 
                                        'base loss':(val_loss/len(val_loader)).item(), 
                                        'peak loss':(val_peak_loss/len(val_loader)).item()}, global_step=i+1)
        writer.add_scalar("lr", curr_lr, global_step=i+1)

        # save model for every best loss epoch
        if not best_loss:
            best_loss = val_step_loss
        elif val_step_loss < best_loss:
            torch.save(model, f'../models/kappa2d_e{i+1}.pth')
            print(f"saved best loss model at epoch = {i+1}!")
            best_loss = val_step_loss
            best_epoch = i+1
    
    print(f"best epoch number is {best_epoch}.")
    writer.close()


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train U2Net')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_1024_2d', type=str, help='data directory')
    parser.add_argument("--cpu", default=32, type=int, help='number of cpu cores to use')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("-e", "--epochs", default=256, type=int, help='number of total epochs to train')
    parser.add_argument("-b", "--batch-size", default=32, type=int, help='batch size')
    parser.add_argument("--lr", default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--ks", default='off', type=str, choices=['off', 'add', 'only'], help='KS93 deconvolution (no KS, KS as an extra channel, no shear and KS only)')
    parser.add_argument("--wiener", default='off', type=str, choices=['off', 'add', 'only'], help='Wiener reconstruction')
    parser.add_argument("--sparse", default='off', type=str, choices=['off', 'add', 'only'], help='sparse reconstruction')
    parser.add_argument("--mcalens", default='off', type=str, choices=['off', 'add', 'only'], help='MCALens reconstruction')
    parser.add_argument("--loss-mode", default='native', type=str, choices=['native', 'gaus'], help='loss function mode')
    parser.add_argument("--loss-fn", default='Huber', type=str, choices=['MSE', 'Huber'], help='loss function: MSE or Huberloss')
    parser.add_argument("--huber-delta", default=50, type=float, help='delta value for Huberloss')
    parser.add_argument("--weight-decay", default=1e-2, type=float, help='weight decay for AdamW optimizer')
    parser.add_argument("--param-count", default=False, action='store_true', help='show model parameter count summary')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    table = PrettyTable(["Arguments", "Values"])
    table.add_row(['start_time', datetime.datetime.now()])
    for arg in vars(args):
        table.add_row([arg, getattr(args, arg)])
    print(table)

    # define training device (cpu/gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device =', device)

    main(args)