import os
import torchvision
import torch
import numpy as np
from numpy import quantile as quant
from PIL import Image


class visualizer():
    """
    Implements helper functions to save losses, logits, networks and intermediate visuals
    """
    def __init__(self, opt):
        folder_losses = os.path.join(opt.checkpoints_dir, opt.exp_name, "losses")

        folder_networks = os.path.join(opt.checkpoints_dir, opt.exp_name, "models")
        if opt.phase == "train":
            folder_images = os.path.join(opt.checkpoints_dir, opt.exp_name, "images")
        else:
            folder_images = os.path.join(opt.checkpoints_dir, opt.exp_name, "evaluation")
        self.losses_saver = losses_saver(folder_losses, opt.continue_epoch)
        self.image_saver = image_saver(folder_images, opt.no_masks, opt.phase, opt.continue_epoch)
        self.network_saver = network_saver(folder_networks, opt.no_EMA)

    def track_losses_logits(self, logits, losses):
        self.losses_saver.track(logits, losses)

    def save_losses_logits(self, epoch):
        self.losses_saver.save(epoch)

    def save_batch(self, fake, epoch, i=None):
        self.image_saver.save(fake, epoch, i)

    def save_networks(self, netG, netD, netEMA ,epoch):
        self.network_saver.save(netG, netD, netEMA, epoch)


class losses_saver():
    def __init__(self, folder_losses, continue_epoch):
        self.folder_losses = folder_losses
        os.makedirs(folder_losses, exist_ok=True)
        self.freq_smooth = 50
        self.logits, self.losses = dict(), dict()
        self.cur_estimates, self.cur_count = dict(), dict()
        self.cur_log = dict()
        self.counter = 0
        if continue_epoch > 0:
            self.logits, self.losses = self.load(["logits", "losses"])

    def load(self, lst):
        ans = list()
        for item in lst:
            cur_dict = dict()
            with open(os.path.join(self.folder_losses, item+".csv"), "r") as f:
                cur_file = f.readlines()
            for line in cur_file:
                elements = line.replace("\n", "").split(",")
                cur_dict[elements[0]] = elements[1:]
            ans.append(cur_dict)
        return ans

    def collect_logits(self, logits):
        ans, cou = 0, 0
        for item in logits:
            for logit in logits[item]:
                ans += logit.detach().cpu().numpy().mean()
                cou += 1
        return ans / cou

    def track(self, logits, losses):
        # --- losses --- #
        for loss_type in losses:
            for loss_part in losses[loss_type]:
                self.cur_estimates[loss_type+"__"+loss_part] = self.cur_estimates.get(loss_type+"__"+loss_part, 0) + \
                                                        float(losses[loss_type][loss_part].detach().cpu())
                self.cur_count[loss_type+"__"+loss_part] = self.cur_count.get(loss_type+"__"+loss_part, 0) + 1
        if self.counter % self.freq_smooth == self.freq_smooth - 1:
            for loss in self.cur_estimates:
                self.losses[loss] = self.losses.get(loss, []) + [str(self.cur_estimates[loss] / self.cur_count[loss])]
            self.cur_estimates, self.cur_count = dict(), dict()

        # --- logits --- #
        for item in ["Dreal", "Dfake"]:
            self.cur_log[item] = self.cur_log.get(item, []) + [self.collect_logits(logits[item])]
        if self.counter % self.freq_smooth == self.freq_smooth - 1:
            for item in ["Dreal", "Dfake"]:
                self.logits[item+".1"] = self.logits.get(item+".1", []) + [str(quant(self.cur_log[item], 0.1))]
                self.logits[item+".5"] = self.logits.get(item+".5", []) + [str(quant(self.cur_log[item], 0.5))]
                self.logits[item+".9"] = self.logits.get(item+".9", []) + [str(quant(self.cur_log[item], 0.9))]
            self.cur_estimates_log = dict()
        self.counter += 1

    def save(self, epoch):
        # --- losses --- #
        with open(os.path.join(self.folder_losses, "losses.csv"), "w") as f:
            for item in self.losses:
                f.writelines([item, ",", ",".join(self.losses[item]), "\n"])
        # --- logits --- #
        with open(os.path.join(self.folder_losses, "logits.csv"), "w") as f:
            for item in self.logits:
                f.writelines([item, ",", ",".join(self.logits[item]), "\n"])


class image_saver():
    def __init__(self, folder_images, no_masks, phase, continue_epoch):
        self.folder_images = folder_images
        self.no_masks = no_masks
        self.phase = phase
        self.ext = ".png" if self.phase == "test" else ".jpg"
        if phase == "test":
            os.makedirs(os.path.join(folder_images, str(continue_epoch)), exist_ok=True)
        else:
            os.makedirs(folder_images, exist_ok=True)

    def save(self, fake, epoch, i=None):
        epoch = str(epoch) + "/" + str(i) if i is not None else str(epoch)

        images = (fake["images"][-1] + 1) / 2
        torchvision.utils.save_image(images, os.path.join(self.folder_images, epoch+self.ext))
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")

    def paint_mask(self, masks):
        ans = torch.zeros((masks.shape[0], 3, masks.shape[2], masks.shape[3]))
        for i in range(masks.shape[0]):
            max_mask = np.argmax(masks[i].cpu().detach().numpy(), axis=0)
            im = Image.fromarray(max_mask.astype(np.uint8)).convert("P")
            im.putpalette(PALETTE)
            im = im.convert("RGB")
            ans[i] = torch.Tensor(np.array(im)).permute(2, 0, 1)
        return ans


class network_saver():
    def __init__(self, folder_networks, no_EMA):
        self.folder_networks = folder_networks
        self.no_EMA = no_EMA
        os.makedirs(folder_networks, exist_ok=True)

    def save(self, netG, netD, netEMA, epoch):
        torch.save(netG.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_G.pth"))
        torch.save(netD.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_D.pth"))
        if not self.no_EMA:
            torch.save(netEMA.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_G_EMA.pth"))
        with open(os.path.join(self.folder_networks, "latest_epoch.txt"), "w") as f:
            f.write(str(epoch))


# --- palette to colorize the mask visualizations --- #
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0,
                191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128,
                64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25,
                26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34,
                35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43,
                44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52,
                53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61,
                62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
                71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79,
                80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88,
                89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97,
                98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105,
                105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112,
                113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120,
                120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127,
                127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134,
                135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142,
                142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149,
                149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
                157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164,
                164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171,
                171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178,
                179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186,
                186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193,
                193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200,
                201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208,
                208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215,
                215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222,
                223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230,
                230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237,
                237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244,
                245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252,
                252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
