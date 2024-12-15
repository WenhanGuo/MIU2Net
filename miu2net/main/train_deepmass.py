from model import UNetDeepMass
import torch
from torch import nn
import transforms as T
from torch.utils.data import DataLoader
from my_dataset import ImageDataset
from loss_functions import loss_fn_selector
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import math
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
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# U2Net total trainable params = 43994893

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


def main(args):

    # prepare train and validation datasets; augment data with flip & rotations; add noise
    train_data = ImageDataset(catalog=os.path.join(args.dir, 'train.ecsv'), 
                              args=args, 
                              transforms=T.Compose([
                                  T.ToTensor(), 
                                  T.AddGaussianNoise(args), 
                                  T.RandomHorizontalFlip(prob=0.5), 
                                  T.RandomVerticalFlip(prob=0.5), 
                                  T.DiscreteRotation(angles=[0, 90, 180, 270]), 
                                  T.RandomCrop(size=args.crop), 
                                  T.Wiener(args), 
                                  T.Resize(size=args.resize), 
                                  T.AddStarMask(args)])
                              )
    val_data = ImageDataset(catalog=os.path.join(args.dir, 'validation.ecsv'), 
                            args=args, 
                            transforms=T.Compose([
                                T.ToTensor(), 
                                T.ReducedShear(args), 
                                T.AddGaussianNoise(args), 
                                T.RandomCrop(size=args.crop), 
                                T.Wiener(args), 
                                T.Resize(size=args.resize), 
                                T.AddStarMask(args)])
                            )
    # prepare train and validation dataloaders
    loader_args = dict(batch_size=args.batch_size, num_workers=args.cpu, pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=True, drop_last=True, **loader_args)

    # initialize U2Net model
    assert args.wiener == 'only'
    model = UNetDeepMass()
    
    if args.load:
        print(f'initializing model using {args.load}.pth')
        state_dict = torch.load('../models/'+args.load+'.pth', map_location=device)
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.param_count == True:
        count_parameters(model)
    # data parallel training on multiple GPUs (restrained by cuda visible devices)
    if len(args.gpu_ids) > 1:
        model.to(args.gpu_ids[0])
        model = nn.DataParallel(model, args.gpu_ids)  # multi-GPUs
    else:
        model.to(device, memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    # setting loss function
    assert args.spac_loss == 'huber'
    loss_fn = loss_fn_selector(args, device)
    loss_fn = loss_fn.to(device)

    # setting optimizer & lr scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, 
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=3)
    # Use torch.cuda.amp for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == True else None

    # use tensorboard to visualize computation
    writer = SummaryWriter('../logs/deepmass')
    # delete existing tensorboard logs
    shutil.rmtree('../logs/deepmass')
    os.mkdir('../logs/deepmass')

    # begin training
    total_train_step = 0
    best_loss = False
    for i in range(args.epochs):
        print(f"--------------------------Starting epoch {i+1}--------------------------")
        # training steps
        model.train()
        for image, target in train_loader:
            image = image.to(device, memory_format=torch.channels_last)
            # image shape: (batchsize, 3 or 4, 256, 256)
            target = target.to(device, memory_format=torch.channels_last)
            # target shape: (batchsize, 1, 256, 256)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                train_step_loss = loss_fn(output, target)

            # optimizer step
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(train_step_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                train_step_loss.backward()
                optimizer.step()

            total_train_step += 1
            # print train losses args.print_freq times per epoch
            if total_train_step % (len(train_loader) // args.print_freq) == 0:
                print(f"train step: {total_train_step}, loss: {train_step_loss.item():.3}")

            lr_scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

        # validation steps
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, target in val_loader:
                image = image.to(device, memory_format=torch.channels_last)
                target = target.to(device, memory_format=torch.channels_last)
                output = model(image)
                val_loss += loss_fn(output, target).item() / len(val_loader)

        # printing epoch stats & writing to tensorboard
        print(f"epoch training loss = {train_step_loss:.4}", f"LR = {curr_lr:.3}")
        print(f"avg validation loss = {val_loss:.4}")
        writer.add_scalar("lr", curr_lr, global_step=i+1)
        writer.add_scalar("train_loss", train_step_loss.item(), global_step=i+1)
        writer.add_scalar("val_loss", val_loss, global_step=i+1)
            
        # save model for every best loss epoch
        if not best_loss:
            best_loss = val_loss
        elif val_loss < best_loss:
            state_dict = model.state_dict()
            torch.save(state_dict, f'../models/deepmass_e{i+1}.pth')
            print(f"saved best loss model at epoch = {i+1}!")
            best_loss = val_loss
            best_epoch = i+1
    
    print(f"best epoch number is {best_epoch}.")
    writer.close()


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train DeepMass')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_1024_2d', type=str, help='data directory')
    parser.add_argument("--gpu-ids", default='6', type=str, help='gpu id; multiple gpu use comma; e.g. 0,1,2')
    parser.add_argument("--cpu", default=32, type=int, help='number of cpu cores to use')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("-e", "--epochs", default=512, type=int, help='number of total epochs to train')
    parser.add_argument("-b", "--batch-size", default=64, type=int, help='batch size')
    parser.add_argument("--lr", default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--crop", default=512, type=int, help='crop 1024x1024 kappa to this size')
    parser.add_argument("--resize", default=256, type=int, help='downsample kappa to this size')
    parser.add_argument("--load", default=False, type=str, help='whether to load a pre-trained .pth model')
    parser.add_argument("--mixed-precision", default=False, action='store_true', help='Use torch.cuda.amp for mixed precision training')
    parser.add_argument("--noise-seed", default=0, type=int, help='how many noise realizations for each training image; 0 for new realization every time')

    parser.add_argument("--reduced-shear", default=False, action='store_true', help='use reduced shear (g) instead of shear (gamma)')
    parser.add_argument("--mask-frac", default=0, type=float, help='randomly mask this fraction of pixels')
    parser.add_argument("--rand-mask-frac", default=False, action='store_true', help='if true, then randomize 0% - mask-frac% masked pixels; otherwise always generate mask-frac%')
    parser.add_argument("--ks", default='off', type=str, choices=['off', 'add', 'only'], help='KS93 deconvolution (no KS, KS as an extra channel, no shear and KS only)')
    parser.add_argument("--wiener", default='off', type=str, choices=['off', 'add', 'only'], help='Wiener reconstruction')
    parser.add_argument("--sparse", default='off', type=str, choices=['off', 'add', 'only'], help='sparse reconstruction')
    parser.add_argument("--mcalens", default='off', type=str, choices=['off', 'add', 'only'], help='MCALens reconstruction')
    parser.add_argument("--cosmo2", default=False, action='store_true', help='if using cosmology2')

    parser.add_argument("--spac-loss", default='huber', type=str, choices=['huber', 'l1', 'ssim', 'ms-ssim', 'charbonnier', 'huber-mean'], help='spatial domain loss function')
    parser.add_argument("--freq-loss", default=None, type=str, choices=['freq', 'freq1d'], help='frequency domain loss function')
    parser.add_argument("--f1d-radius", default=16, type=int, help='max radius for freq1d radial power spectrum averaging loss')
    parser.add_argument("--alpha", default=1.0, type=float, help='weight for spatial term in loss function')
    parser.add_argument("--beta", default=10.0, type=float, help='weight for frequency term in loss function')
    parser.add_argument("--wiener-res", default=False, action='store_true', help='if the target is true - wiener')
    parser.add_argument("--assemble-mode", default='1x1conv', type=str, choices=['1x1conv', 'laplacian_pyr'], help='experimental feature')
    parser.add_argument("--huber-delta", default=50.0, type=float, help='delta value for Huberloss')
    parser.add_argument("--weight-decay", default=0, type=float, help='weight decay for AdamW optimizer')
    parser.add_argument("--param-count", default=True, help='show model parameter count summary')
    parser.add_argument("--print-freq", default=3, type=int, help='print train loss how many times for each epoch')

    parser.add_argument("--save-noisy-shear", default=False, action='store_true', help='write shear with added gaussian noise to disk')
    parser.add_argument("--save-noisy-shear-dir", default='/share/lirui/Wenhan/WL/kappa_map/result/noisy_shear', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    table = PrettyTable(["Arguments", "Values"])
    table.add_row(['start_time', datetime.datetime.now()])
    for arg in vars(args):
        table.add_row([arg, getattr(args, arg)])
    print(table)

    # define training device (cpu/gpu)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
    device = torch.device(f'cuda:{args.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    main(args)