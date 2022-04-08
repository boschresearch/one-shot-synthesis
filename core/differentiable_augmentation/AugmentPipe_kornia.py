import torch
import kornia
import random
from torchvision import transforms as TR
import torch.nn.functional as F


class AugmentPipe_kornia(torch.nn.Module):
    def __init__(self, prob, no_masks):
        super().__init__()
        self.prob = prob
        self.no_masks = no_masks

    def forward(self, batch):
        x = batch["images"]
        if not self.no_masks:
            raise ValueError("Kornia augmentations without --no_masks regime is not supported")

        ref = x
        sh = x[-1].shape
        x = combine_fakes(x)

        if random.random() < self.prob/2:
            tr = kornia.augmentation.RandomCrop(size=(sh[2], sh[3]), same_on_batch=True)
            for i in range(sh[0]):
                x[i] = torch.nn.functional.interpolate(x[i], size=(2*sh[2], 2*sh[3]), mode="bilinear")
                x[i] = tr(x[i])

        if random.random() < self.prob:
            if random.random() < 0.5:
                r = random.random() * 0.25 + 0.75
                tr = kornia.augmentation.RandomCrop(size=(int(sh[2]*r), int(sh[3]*r)), same_on_batch=True)
                for i in range(sh[0]):
                    x[i] = tr(x[i])
                    x[i] = torch.nn.functional.interpolate(x[i], size=(sh[2], sh[3]), mode="bilinear")
            else:
                tr = kornia.augmentation.RandomRotation(degrees=8, same_on_batch=True)
                for i in range(sh[0]):
                    x[i] = tr(x[i])
                tr = kornia.augmentation.CenterCrop(size=(sh[2]*0.80, sh[3]*0.80))
                for i in range(sh[0]):
                    x[i] = tr(x[i])
                    x[i] = torch.nn.functional.interpolate(x[i], size=(sh[2], sh[3]), mode="bilinear")

        if random.random() < self.prob:
            tr = kornia.augmentation.RandomHorizontalFlip(p=1.0, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])

        if random.random() < self.prob:
            for i in range(sh[0]):
                x[i] = translate_v_fake(x[i], fraction=(0.05, 0.3))
        if random.random() < self.prob:
            for i in range(sh[0]):
                x[i] = translate_h_fake(x[i], fraction=(0.05, 0.3))

        x = detach_fakes(x, ref)

        batch["image"] = x
        return batch


def combine_fakes(inp):
    sh = inp[-1].shape
    ans = list()
    for i in range(sh[0]):
        cur = torch.zeros_like(inp[-1][0, :, :, :]).repeat(len(inp), 1, 1, 1)
        for j in range(len(inp)):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0), size=(sh[2], sh[3]),
                                                              mode="bilinear")
        ans.append(cur)
    return ans


def detach_fakes(inp, ref):
    ans = list()
    sh = ref[-1].shape
    for i in range(len(ref)):
        cur = torch.zeros_like(ref[i])
        for j in range(sh[0]):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0),
                                                              size=(ref[i].shape[2], ref[i].shape[3]),
                                                              mode="bilinear")
        ans.append(cur)
    return ans


class myRandomResizedCrop(TR.RandomResizedCrop):
    def __init__(self, size=256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), ):
        super(myRandomResizedCrop, self).__init__(size, scale, ratio)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return TR.functional.resized_crop(img, i, j, h, w, (img.size[1], img.size[0]), self.interpolation)


def translate_v_fake(x, fraction):
    margin = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_up = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    left, right = 0, width
    if direct_up:
        top, bottom = 0, int(height * margin)
    else:
        top, bottom = height - int(height * margin), height
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (2,))
    if not direct_up:
        x[:, :, 0:height - int(height * margin), :] = x[:, :, int(height * margin):height, :].clone()
        x[:, :, height - int(height * margin):, :] = im_to_paste
    else:
        x[:, :, int(height * margin):height, :] = x[:, :, 0:height - int(height * margin), :].clone()
        x[:, :, :int(height * margin), :] = im_to_paste
    return x


def translate_h_fake(x, fraction):
    margin = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_left = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    top, bottom = 0, height
    if direct_left:
        left, right = 0, int(width * margin)
    else:
        left, right = width - int(width * margin), width
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (3,))
    if not direct_left:
        x[:, :, :, 0:width - int(width * margin)] = x[:, :, :, int(width * margin):width].clone()
        x[:, :, :, width - int(width * margin):] = im_to_paste
    else:
        x[:, :, :, int(width * margin):width] = x[:, :, :, 0:width - int(width * margin)].clone()
        x[:, :, :, :int(width * margin)] = im_to_paste
    return x