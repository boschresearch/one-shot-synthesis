"""
This file computes metrics for a chosen checkpoint. By default, it computes SIFID (at lowest InceptionV3 scale),
LPIPS diversity, LPIPS distance to training data, mIoU (in case segmentation masks are used).
The results are saved at /${checkpoints_dir}/${exp_name}/metrics/
For SIFID, LPIPS_to_train, mIoU, and segm accuracy, the metrics are computed per each image.
LPIPS, mIoU and segm_accuracy are also computed for the whole dataset.
"""


import os
import argparse
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from metrics import SIFID, LPIPS, LPIPS_to_train, mIoU


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--epoch', type=str, required=True)
parser.add_argument('--sifid_all_layers', action='store_true')
args = parser.parse_args()

print("--- Computing metrics for job %s at epoch %s ---" %(args.exp_name, args.epoch))


def convert_sifid_dict(names_fake_image, sifid):
    ans = dict()
    for i, item in enumerate(names_fake_image):
        ans[item] = sifid[i]
    return ans


exp_folder = os.path.join(args.checkpoints_dir, args.exp_name, "evaluation", args.epoch)
if not os.path.isdir(exp_folder):
    raise ValueError("Generated images not found. Run the test script first. (%s)" % (exp_folder))


# --- Read options file from checkpoint --- #
file_name = os.path.join(args.checkpoints_dir, args.exp_name, "opt.pkl")
new_opt = pickle.load(open(file_name, 'rb'))
no_masks = getattr(new_opt, "no_masks")
dataroot = getattr(new_opt, "dataroot")
dataset_name = getattr(new_opt, "dataset_name")
path_real_images = os.path.join(dataroot, dataset_name, "image")
if not no_masks:
    raise NotImplementedError("w/o --no_masks is not implemented in this release")

# --- Prepare files and images to compute metrics --- #
names_real_image = sorted(os.listdir(path_real_images))
if not no_masks:
    raise NotImplementedError("w/o --no_masks is not implemented in this release")
names_fake = sorted(os.listdir(os.path.join(exp_folder)))
names_fake_image = [item for item in names_fake if "mask" not in item]
if not no_masks:
    raise NotImplementedError("w/o --no_masks is not implemented in this release")
list_real_image, list_fake_image = list(), list()
for i in range(len(names_fake_image)):
    im = (Image.open(os.path.join(exp_folder, names_fake_image[i])).convert("RGB"))
    list_fake_image += [im]
im_res = (ToTensor()(list_fake_image[0]).shape[2], ToTensor()(list_fake_image[0]).shape[1])
for i in range(len(names_real_image)):
    im = (Image.open(os.path.join(path_real_images, names_real_image[i])).convert("RGB"))
    list_real_image += [im.resize(im_res, Image.BILINEAR)]

# --- Compute the metrics --- #
with torch.no_grad():
    sifid1, sifid2, sifid3, sifid4 = SIFID(list_real_image, list_fake_image, args.sifid_all_layers)
    lpips = LPIPS(list_fake_image)
    dist_to_tr, dist_to_tr_byimage = LPIPS_to_train(list_real_image, list_fake_image, names_fake_image)
if not no_masks:
    raise NotImplementedError("w/o --no_masks is not implemented in this release")

# --- Save the metrics under .${exp_name}/metrics --- #
save_fld = os.path.join(args.checkpoints_dir, args.exp_name, "metrics")
os.makedirs(save_fld, exist_ok=True)

sifid1 = convert_sifid_dict(names_fake_image, sifid1)
np.save(os.path.join(save_fld, str(args.epoch))+"SIFID1", sifid1)
if sifid2 is not None:
    sifid2 = convert_sifid_dict(names_fake_image, sifid2)
    np.save(os.path.join(save_fld, str(args.epoch))+"SIFID2", sifid2)
if sifid3 is not None:
    sifid3 = convert_sifid_dict(names_fake_image, sifid3)
    np.save(os.path.join(save_fld, str(args.epoch))+"SIFID3", sifid3)
if sifid4 is not None:
    sifid4 = convert_sifid_dict(names_fake_image, sifid4)
    np.save(os.path.join(save_fld, str(args.epoch))+"SIFID4", sifid4)
np.save(os.path.join(save_fld, str(args.epoch))+"lpips", lpips.cpu())
np.save(os.path.join(save_fld, str(args.epoch))+"dist_to_tr", dist_to_tr.cpu())
np.save(os.path.join(save_fld, str(args.epoch))+"dist_to_tr_byimage", dist_to_tr_byimage)
if not no_masks:
    raise NotImplementedError("w/o --no_masks is not implemented in this release")
print("--- Saved metrics at %s ---" % (save_fld))
