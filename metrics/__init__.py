import torch
from torchvision.transforms import ToTensor
from .SIFID.sifid_score import calculate_sifid_given_paths
from .FID.tests_with_FID import calculate_fid_given_paths
from .mIoU.main import compute_miou
from .LPIPS.models import PerceptualLoss

p_model = PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

def SIFID(list_real_image, list_fake_image, sifid_all_layers):
    """
    When learning from a Single Image, compute SIFID from fake images to the real image.
    In case of multiple training images, compute FID between full fake and real sets.
    By default, compute only sifid at lowest InceptionV3 layer (sifid1)
    """
    if len(list_real_image):
        sifid1 = calculate_sifid_given_paths(list_real_image, list_fake_image, 1, True, 64)
        if not sifid_all_layers:
            return sifid1, None, None, None
        sifid2 = calculate_sifid_given_paths(list_real_image, list_fake_image, 1, True, 192)
        sifid3 = calculate_sifid_given_paths(list_real_image, list_fake_image, 1, True, 768)
        sifid4 = calculate_sifid_given_paths(list_real_image, list_fake_image, 1, True, 2048)
        return sifid1, sifid2, sifid3, sifid4
    else:
        sifid1 = calculate_fid_given_paths([list_real_image, list_fake_image], 10, True, 64)
        if not sifid_all_layers:
            return sifid1, None, None, None
        sifid2 = calculate_fid_given_paths([list_real_image, list_fake_image], 10, True, 192)
        sifid3 = calculate_fid_given_paths([list_real_image, list_fake_image], 10, True, 768)
        sifid4 = calculate_fid_given_paths([list_real_image, list_fake_image], 10, True, 2048)
        return sifid1, sifid2, sifid3, sifid4


def LPIPS(list_fake_image):
    """
    Compute average LPIPS between pairs of fake images
    """
    dist_diversity = 0
    count = 0
    lst_im = list()
    # --- unpack images --- #
    for i in range(len(list_fake_image)):
        lst_im.append(ToTensor()(list_fake_image[i]).unsqueeze(0))
    # --- compute LPIPS between pairs of images --- #
    for i in range(len(lst_im))[:100]:
        for j in range(i + 1, len(lst_im))[:100]:
            dist_diversity += p_model.forward(lst_im[i], lst_im[j])
            count += 1
    return dist_diversity/count


def LPIPS_to_train(list_real_image, list_fake_image, names_fake_image):
    """
    For each fake image find the LPIPS to the closest training image
    """
    dist_to_real_dict = dict()
    ans1 = 0
    count = 0
    lst_real, list_fake = list(), list()
    # --- unpack images --- #
    for i in range(len(list_fake_image)):
        list_fake.append(ToTensor()(list_fake_image[i]).unsqueeze(0))
    for i in range(len(list_real_image)):
        lst_real.append(ToTensor()(list_real_image[i]).unsqueeze(0))
    # --- compute average minimum LPIPS from a fake image to real images --- #
    for i in range(len(list_fake)):
        tens_im1 = list_fake[i]
        cur_ans = list()
        for j in range(len(lst_real)):
            tens_im2 = lst_real[j]
            dist_to_real = p_model.forward(tens_im1, tens_im2)
            cur_ans.append(dist_to_real)
        cur_min = torch.min(torch.Tensor(cur_ans))
        dist_to_real_dict[names_fake_image[i]] = float(cur_min.detach().cpu().item())
        ans1 += cur_min
        count += 1
    ans = ans1 / count
    return ans, dist_to_real_dict

def mIoU(path_real_images, names_real_image, path_real_masks, names_real_masks,
             exp_folder, names_fake_image, names_fake_masks, im_res):
    """
    Train a simple UNet on fake (real) images&masks, test on real (fake) images&masks.
    Report mIoU and segmentation accuracy for the whole sets (fake->real and  real->fake) as well as
    individual scores for each fake image
    """
    metrics_tensor, results, results_acc = compute_miou(path_real_images, names_real_image, path_real_masks, names_real_masks,
                                                        exp_folder, names_fake_image, names_fake_masks, im_res)
    return metrics_tensor, results, results_acc
