"""
This code implements a simple UNet to compute the mIoU metric evaluating the quality of segmentation masks.
This code is adopted from usuyama/pytorch-unet.
"""


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from .loss import dice_loss
from .dataset import SimDataset
from .unet import ResNetUNet


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    ###outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    SMOOTH = 1e-6
    outputs = torch.nn.functional.one_hot(outputs)
    labels = torch.nn.functional.one_hot(labels)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    #print(iou.shape, thresholded.shape)
    return thresholded


def train_model(model, optimizer, scheduler, dataloader_train, num_epochs=500):
    model.train()

    for epoch, batch in enumerate(dataloader_train):
        if epoch >= num_epochs:
            break
        inputs, labels = batch
        scheduler.step()
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        # zero the parameter gradients
        optimizer.zero_grad()
        metrics = defaultdict(float)
        # forward
        # track history if only in train
        outputs = model(inputs)
        loss = calc_loss(outputs, labels, metrics)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
    return model

# -------------------------------------------------------------------


def compute_miou(path_real_images, names_real_image, path_real_masks, names_real_masks,
                                        exp_folder, names_fake_image, names_fake_masks, im_res):
    train_set = SimDataset(path_real_images, names_real_image, path_real_masks, names_real_masks, im_res, real=True)
    num_ch = train_set.num_mask_channels
    val_set = SimDataset(exp_folder, names_fake_image, exp_folder, names_fake_masks, im_res, real=False, num_ch=num_ch)
    image_datasets = {'train': train_set, 'val': val_set}
    batch_size = 5
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(n_class=num_ch)
    model = model.to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=1.0)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders["val"], 500)

    model.eval()   # Set model to the evaluation mode
    all_corr, sum_corr, cur_iou, countt = 0, 0, 0, 0
    for i, batch in enumerate(dataloaders["train"]):
        if i > 100:
            break
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Predict
        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        pred = pred.data
        pred1 = torch.argmax(pred, dim=1)
        pred2 = torch.argmax(labels, dim=1)
        correct = ((pred1 == pred2)*1).sum()
        sum_corr += correct
        all_corr += torch.numel(pred1)
        cur_iou += iou_pytorch(pred1, pred2).mean()
        countt += 1
    metrics_tensor = np.array([-1.0, -1.0, -1.0, -1.0])
    metrics_tensor[0] = sum_corr / all_corr
    metrics_tensor[1] = cur_iou / countt

    # HERE TRAINING on real

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = ResNetUNet(n_class=num_ch)
    model1 = model1.to(device)
    model1.train()

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=1.0)

    model1 = train_model(model1, optimizer_ft, exp_lr_scheduler, dataloaders["train"], 500)
    model1.eval()   # Set model to the evaluation mode
    all_corr, sum_corr, cur_iou, countt = 0, 0, 0, 0
    for i, batch in enumerate(dataloaders["val"]):
        if i > 100:
            break
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Predict
        pred = model1(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        pred = pred.data

        pred1 = torch.argmax(pred, dim=1)
        pred2 = torch.argmax(labels, dim=1)
        correct = ((pred1 == pred2)*1).sum()
        sum_corr += correct
        all_corr += torch.numel(pred1)

        cur_iou += iou_pytorch(pred1, pred2).mean()
        countt += 1

    metrics_tensor[2] = sum_corr / all_corr
    metrics_tensor[3] = cur_iou / countt

    #### metric per image below:

    val_per_fr_set = SimDataset(exp_folder, names_fake_image, exp_folder, names_fake_masks, im_res, real=False, num_ch=num_ch, no_transform=True)
    dataloader_per_frame = DataLoader(val_per_fr_set, batch_size=1, shuffle=False, num_workers=0)

    results = dict()
    results_acc = dict()

    for i, batch in enumerate(dataloader_per_frame):
        if i >= len(val_per_fr_set.masks):
            break
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Predict
        pred = model1(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        pred = pred.data

        pred1 = torch.argmax(pred, dim=1)
        pred2 = torch.argmax(labels, dim=1)
        correct = ((pred1 == pred2)*1).sum()
        acc = correct / torch.numel(pred1)
        results_acc[val_per_fr_set.frames[i]] = acc.detach().cpu().numpy()
        cur_iou = iou_pytorch(pred1, pred2).mean()
        results[val_per_fr_set.frames[i]] = cur_iou.detach().cpu().numpy()

    return metrics_tensor, results, results_acc
