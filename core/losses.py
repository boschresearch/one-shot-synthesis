import torch
import torch.nn.functional as F


class losses_computer():
    def __init__(self, opt, num_blocks):
        """
        The class implementing the loss computations
        """
        self.loss_function = self.get_loss_function(opt.loss_mode)
        self.no_masks = opt.no_masks
        self.no_DR = opt.no_DR
        self.lambdas = {"content": 0.5 / num_blocks,
                        "layout": 0.5 / num_blocks,
                        "low-level": 1.0 / num_blocks,
                        "DR": opt.lambda_DR}

    def get_loss_function(self, loss_mode):
        if loss_mode == "wgan":
            return wgan_loss
        elif loss_mode == "hinge":
            return hinge_loss
        elif loss_mode == "bce":
            return bce_loss
        else:
            raise ValueError('Unexpected loss_mode {}'.format(mode))

    def content_segm_loss(self, out_d, data, real, forD):
        """
        The multi-class cross-entropy loss used in the content masked attention
        """
        mask = data["masks"]
        mask_ch = mask.shape[1]
        if real:
            ground_t = torch.arange(mask_ch).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ground_t = ground_t.repeat(1, 1, out_d.shape[2], out_d.shape[3])
            ground_t = ground_t.repeat_interleave(mask.shape[0], dim=0)[:, 0, :, :]
        else:  # fake
            ground_t = torch.ones_like(out_d)[:, 0, :, :] * mask_ch
        weights = torch.cat((1 / (torch.sum(mask.detach(), dim=(0, 2, 3))), torch.Tensor([1.0]).to(out_d.device)))
        weights[weights == float('inf')] = 0
        loss = F.cross_entropy(out_d, ground_t.long().to(out_d.device), weight=weights.to(out_d.device))
        return loss

    def diversity_regularization(self, fake):
        """
        The diversity regularization applied in the feature space of the generator
        """
        loss = torch.nn.L1Loss()
        ans = 0
        for i in range(len(fake)):
            for k in range(fake[i].shape[0]):
                for m in range(k + 1, fake[i].shape[0]):
                    ans += -loss(fake[i][k], fake[i][m])
        return ans * 2 / (len(fake) * (len(fake) - 1))

    def balance_losses(self, losses):
        """
        Multiply each loss part with its lambda
        """
        for item in losses:
            if item in self.lambdas.keys():
                losses[item] = losses[item] * self.lambdas[item]
        return losses

    def __call__(self, out_d, data, real, forD):
        losses = {}
        # --- adversarial loss ---#
        for item in out_d:
            for i in range(len(out_d[item])):
                if item == "content" and not self.no_masks:
                    raise NotImplementedError("w/o --no_masks is not implemented in this release")
                else:
                    losses[item] = losses.get(item, 0) + self.loss_function(out_d[item][i], real, forD)

        # --- diversity regularization ---#
        if not forD and not self.no_DR:
            losses["DR"] = self.diversity_regularization(data["features"])
        losses = self.balance_losses(losses)
        return losses


def wgan_loss(output, real, forD):
    if real and forD:
        ans = -output.mean()
    elif not real and forD:
        ans = output.mean()
    elif real and not forD:
        ans = -output.mean()
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    #print(real, forD, ans)
    return ans


def hinge_loss(output, real, forD):
    if real and forD:
        minval = torch.min(output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif not real and forD:
        minval = torch.min(-output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif real and not forD:
        ans = -torch.mean(output)
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    return ans


def bce_loss(output, real, forD, no_aggr=False):
    target_tensor = get_target_tensor(output, real).to(output.device)
    ans = F.binary_cross_entropy_with_logits(output, target_tensor, reduction=("mean" if not no_aggr else "none"))
    return ans


def get_target_tensor(input, target_is_real):
    if target_is_real:
        real_label_tensor = torch.FloatTensor(1).fill_(1)
        real_label_tensor.requires_grad_(False)
    else:
        real_label_tensor = torch.FloatTensor(1).fill_(0)
        real_label_tensor.requires_grad_(False)
    return real_label_tensor.expand_as(input)


def get_zero_tensor(input):
    zero_tensor = torch.FloatTensor(1).fill_(0)
    zero_tensor.requires_grad_(False)
    return zero_tensor.expand_as(input)
