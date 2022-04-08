import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sp_norm
import numpy as np
import time
import os


def fix_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.exp_name, "progress.txt")
        with open(self.file_name, "a") as log_file:
            log_file.write('--- Started training --- \n')

    def __call__(self, epoch):
        if epoch != 0:
            avg = (time.time() - self.prev_time) / (epoch - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = epoch

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d], avg time:%.3f per epoch \n' % (epoch, self.num_epochs, avg))
        print('[epoch %d/%d], avg time:%.3f per epoch' % (epoch, self.num_epochs, avg))
        return avg


def update_EMA(netEMA, netG, EMA_decay):
    with torch.no_grad():
        for key in netG.state_dict():
            netEMA.state_dict()[key].data.copy_(
                netEMA.state_dict()[key].data * EMA_decay +
                netG.state_dict()[key].data * (1 - EMA_decay)
            )
    return netEMA


def preprocess_real(batch, num_blocks_ll, device):
    # --- Put everything on GPU if needed --- #
    for item in batch:
        batch[item] = batch[item].to(device)
    # --- Create downsampled versions of real images for MSG --- #
    ans = list()
    image = batch["images"]
    ans.append(image)
    for i in range(num_blocks_ll-1):
        image = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        ans.append(image)
    batch["images"] = list(reversed(ans))
    return batch


def sample_noise(noise_dim, batch_size):
    return torch.randn(batch_size, noise_dim, 1, 1)


def to_rgb(in_channels):
    return sp_norm(nn.Conv2d(in_channels, 3, (3, 3), padding=(1, 1), bias=True))


def get_norm_by_name(norm_name, out_channel):
    if norm_name == "batch":
        return nn.BatchNorm2d(out_channel)
    if norm_name == "instance":
        return nn.InstanceNorm2d(out_channel)
    if norm_name == "none":
        return nn.Sequential()
    raise NotImplementedError("The norm name is not recognized %s" % (norm_name))


def from_rgb(out_channels):
    return sp_norm(nn.Conv2d(3, out_channels, (3, 3), padding=(1, 1), bias=True))


def to_decision(out_channel, target_channel):
    return sp_norm(nn.Conv2d(out_channel, target_channel, (1,1)))