import torch
from torch import nn


class FreqLoss(nn.Module):
    def __init__(self, img_size, radius, weight=1, device=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.size = img_size
        self.radius = radius
        self.mask = self.create_mask().to(device)
        self.weight = weight
    
    def create_mask(self):
        s = self.size
        x, y = torch.meshgrid(torch.arange(-s/2, s/2), torch.arange(-s/2, s/2), indexing='ij')
        d = torch.sqrt((x+0.5)**2 + (y+0.5)**2)
        mask = d <= self.radius
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        return mask
    
    def power_spec(self, image):
        ft = torch.fft.fft2(image)
        pspec = torch.abs(ft)**2
        pspec_shifted = torch.fft.fftshift(pspec)
        return pspec_shifted
    
    def forward(self, output, target):
        ps_output = self.power_spec(output) * self.mask
        ps_target = self.power_spec(target) * self.mask
        return self.l1(ps_output, ps_target) * self.weight


class FreqLoss1D(nn.Module):
    def __init__(self, img_size, radius, weight=1, device=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.size = img_size
        self.radius = radius
        self.masks = self.create_masks().to(device)
        self.weight = weight
    
    def create_masks(self):
        s = self.size
        bins = torch.arange(0, self.radius+1.0, 1.0)
        x, y = torch.meshgrid(torch.arange(-s/2, s/2), torch.arange(-s/2, s/2), indexing='ij')
        d = torch.sqrt((x+0.5)**2 + (y+0.5)**2)
        d = d.unsqueeze(0).expand(len(bins)-1, s, s)
        bins_low = bins[:-1].unsqueeze(1).unsqueeze(2).expand(len(bins)-1, s, s)
        bins_upp = bins[1:].unsqueeze(1).unsqueeze(2).expand(len(bins)-1, s, s)
        masks = (d >= bins_low) & (d < bins_upp)
        return masks.unsqueeze(0).unsqueeze(0)
    
    def power_spec(self, image):
        ft = torch.fft.fft2(image)
        pspec = torch.abs(ft)**2
        pspec_shifted = torch.fft.fftshift(pspec)
        return pspec_shifted
    
    def radial_power_spec(self, ps2D):
        # radial average power spectrum
        rad_ps = torch.mean(ps2D.unsqueeze(2) * self.masks, dim=(3, 4))
        # sum across channel dimension
        rad_ps = torch.sum(rad_ps, dim=1)
        return rad_ps
    
    def forward(self, output, target):
        ps_output = self.radial_power_spec(ps2D=self.power_spec(output))
        ps_target = self.radial_power_spec(ps2D=self.power_spec(target))
        return self.l1(ps_output, ps_target) * self.weight


class L1Mod(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.weight = weight
    
    def forward(self, output, target):
        return self.l1(output, target) * self.weight


class HuberMod(nn.Module):
    def __init__(self, delta, mean_penalty=0, weight=1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.mean_penalty = mean_penalty
        self.weight = weight
    
    def forward(self, output, target):
        mean_loss = abs(torch.mean(output) - torch.mean(target)) * self.mean_penalty
        huber_loss = self.huber(output, target)
        return (mean_loss + huber_loss) * self.weight


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9, weight=1):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.weight = weight

    def forward(self, output, target):
        diff = output - target
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss * self.weight


def loss_fn_selector(args, device):
    # a, b (alpha, beta declared in args) set the relative importance of spatial and frequency terms
    # if a = b, default weights in functions will scale each term to be approx equal when training converges
    a, b = args.alpha, args.beta

    if args.spac_loss == 'huber':
        spac_fn = HuberMod(delta=args.huber_delta, weight=1e4*a)
    elif args.spac_loss == 'l1':
        spac_fn = L1Mod(weight=1e3*a)
    elif args.spac_loss == 'charbonnier':
        spac_fn = CharbonnierLoss(weight=1e2*a)
    if args.spac_loss == 'huber-mean':
        spac_fn = HuberMod(delta=args.huber_delta, mean_penalty=0.1, weight=1e4*a)

    if args.freq_loss == 'freq':
        freq_fn = FreqLoss(img_size=args.resize, radius=args.f1d_radius, weight=1*b, device=device)
    elif args.freq_loss == 'freq1d':
        freq_fn = FreqLoss1D(img_size=args.resize, radius=args.f1d_radius, weight=10*b, device=device)

    if args.freq_loss == None:
        return spac_fn
    else:
        return freq_fn, spac_fn