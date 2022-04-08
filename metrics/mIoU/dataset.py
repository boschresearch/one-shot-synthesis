import random
import os
import numpy as np
from torchvision import transforms as TR
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader


class SimDataset(Dataset):
    def __init__(self, fld_im, names_real_image, fld_mask, names_real_masks, im_res, real=True, num_ch=None, no_transform=False):
        self.real = real

        self.frame_path = fld_im
        self.frames = names_real_image
        self.mask_path = fld_mask
        self.masks = names_real_masks
        self.im_res = (im_res[1], im_res[0])
        self.no_transform = no_transform
        if real:
            self.num_mask_channels = self.get_num_mask_channels()
        else:
            self.num_mask_channels = num_ch

        self.transforms = get_transforms(im_res, no_transform)

    def __len__(self):
        return 10000000

    def __getitem__(self, indx):
        idx = indx % len(self.frames)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        img_pil = Image.open("%s/%s" % (self.frame_path, self.frames[idx])).convert("RGB")
        target_size = self.im_res

        res = self.transforms(TR.functional.resize(img_pil, size=target_size)).to("cuda")
        ans = (res - 0.5) * 2

        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

        mask_pil = Image.open("%s/%s" % (self.mask_path, self.masks[idx][:-4] + ".png"))
        mask = self.transforms(TR.functional.resize(mask_pil, size=target_size, interpolation=Image.NEAREST)).to("cuda")
        mask = self.create_mask_channels(mask)  #  mask should be N+1 channels
        return [ans, mask]

    def create_mask_channels(self, mask):
        if (mask.unique() * 256).max() > 20:  # only object and background
            mask = (torch.sum(mask, dim=(0,), keepdim=True) > 0)*1.0
            mask = torch.cat((1 - mask, mask), dim=0)
            return mask
        else:  # background and many objects
            integers = torch.round(mask * 256)
            mask = torch.nn.functional.one_hot(integers.long(), num_classes=self.num_mask_channels).float()[
                0].permute(2, 0, 1)
            return mask

    def get_num_mask_channels(self):
        masks = self.masks
        c = 0
        for item in range(len(masks)):
            im = TR.ToTensor()(Image.open(os.path.join(self.mask_path, masks[item])))
            if (im.unique() * 256).max() > 20:
                c = 2 if 2 > c else c
            else:
                cur = torch.max(torch.round(im * 256))
                c = cur + 1 if cur + 1 > c else c
        return int(c)


def get_transforms(im_res, no_transform):
    prob_augm = 0.3
    tr_list = list()

    if not no_transform:
        TR.RandomApply(
            [TR.RandomResizedCrop(size=im_res, scale=(0.75, 1.0), ratio=(1, 1))],
            p=prob_augm),

        tr_list.append(TR.RandomApply([TR.RandomHorizontalFlip(p=1)], p=prob_augm / 2)),
        tr_list.append(TR.RandomApply([myVerticalTranslation(fraction=(0.05, 0.3))], p=prob_augm)),
        tr_list.append(TR.RandomApply([myHorizontalTranslation(fraction=(0.05, 0.3))], p=prob_augm)),
    tr_list.append(TR.ToTensor())
    return TR.Compose(tr_list)


class myRandomResizedCrop(TR.RandomResizedCrop):
    def __init__(self, size=256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), ):
        super(myRandomResizedCrop, self).__init__(size, scale, ratio)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return TR.functional.resized_crop(img, i, j, h, w, (img.size[1], img.size[0]), self.interpolation)


class myVerticalTranslation(TR.RandomResizedCrop):
    def __init__(self, fraction=(0.05, 0.3)):
        self.fraction = fraction
        super(myVerticalTranslation, self).__init__(size=256)

    def __call__(self, img):
        margin = torch.rand(1) * (self.fraction[1] - self.fraction[0]) + self.fraction[0]
        direct_up = (torch.rand(1) < 0.5)  # up or down
        width, height = img.size
        left, right = 0, width
        shift = -int(height * margin) if direct_up else int(height * margin)
        if direct_up:
            top, bottom = 0, int(height * margin),
        else:
            top, bottom = height - int(height * margin), height
        im_to_paste = ImageOps.flip(img.crop((left, top, right, bottom)))
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, shift))
        if direct_up:
            img.paste(im_to_paste, (0, 0))
        else:
            img.paste(im_to_paste, (0, height - shift))
        return img


class myHorizontalTranslation(TR.RandomResizedCrop):
    def __init__(self, fraction=(0.05, 0.3)):
        self.fraction = fraction
        super(myHorizontalTranslation, self).__init__(size=256)

    def __call__(self, img):
        margin = torch.rand(1) * (self.fraction[1] - self.fraction[0]) + self.fraction[0]
        direct_left = (torch.rand(1) < 0.5)  # up or down
        width, height = img.size
        top, bottom = 0, height
        shift = -int(width * margin) if direct_left else int(width * margin)
        if direct_left:
            left, right = 0, int(width * margin)
        else:
            left, right = width - int(width * margin), width
        im_to_paste = ImageOps.mirror(img.crop((left, top, right, bottom)))
        img = img.transform(img.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
        if direct_left:
            img.paste(im_to_paste, (0, 0))
        else:
            img.paste(im_to_paste, (width - shift, 0))
        return img