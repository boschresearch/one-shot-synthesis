import torch
import torch.nn as nn
import random


class Content_FA(nn.Module):
    def __init__(self, no_mask, prob_FA_con, num_mask_channels=None):
        super(Content_FA, self).__init__()
        self.prob = prob_FA_con
        self.ranges = (0.10, 0.30)
        self.no_mask = no_mask
        if not self.no_mask:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")

    def mix(self, y):
        """
        Randomly swap channels of different instances
        """
        bs = y.shape[0]
        ch = y.shape[1]
        ans = y
        # ---  --- #
        if random.random() < self.prob:
            for i in range(0, bs - 1, 2):
                num_first = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
                perm = torch.randperm(ch)
                ch_first = perm[:num_first]
                ans[i, ch_first, :, :] = y[i + 1, ch_first, :, :].clone()
                ans[i + 1, ch_first, :, :] = y[i, ch_first, :, :].clone()
        return ans

    def drop(self, y):
        """
        Randomly zero out channels
        """
        ch = y.shape[1]
        ans = y
        if random.random() < self.prob:
            num_first = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
            num_second = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
            perm = torch.randperm(ch)
            ch_second = perm[num_first:num_first + num_second]
            ans[:, ch_second, :, :] = 0
        return ans

    def forward(self, y):
        ans = y
        y = self.mix(y)
        y = self.drop(y)
        if not self.no_mask:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        else:
            ans = y
        return ans


class Layout_FA(nn.Module):
    def __init__(self, no_mask, prob):
        super(Layout_FA, self).__init__()
        self.no_mask = no_mask
        self.prob = prob
        self.ranges = (0.10, 0.30)

    def forward(self, y, masks):
        if not self.no_mask:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        else:
            ans = self.func_without_mask(y)
        return ans

    def func_without_mask(self, y):
        """
        If a segmentation mask is not provided, copy-paste rectangles in a random way
        """
        bs = y.shape[0]
        ans = y.clone()
        for i in range(0, bs - 1, 2):
            if random.random() < self.prob:
                x1, x2, y1, y2 = gen_rectangle(ans)
                ans[i, :, x1:x2, y1:y2] = y[i + 1, :, x1:x2, y1:y2].clone()
                ans[i + 1, :, x1:x2, y1:y2] = y[i, :, x1:x2, y1:y2].clone()
        return ans

    def func_with_mask(self, y, mask):
        """
        If a segmentation mask is provided, ensure that the copied areas never cut semantic boundaries
        """
        ans_y = y.clone()
        ans_mask = mask.clone()
        ans_y, ans_mask = self.mix_background(ans_y, ans_mask)
        ans_y, ans_mask = self.swap(ans_y, ans_mask)
        ans_y, ans_mask = self.move_objects(ans_y, ans_mask)
        return ans_y

    def mix_background(self, y, mask):
        """
        Copy-paste areas of background onto other background areas
        """
        for i in range(0, y.shape[0]):
            if random.random() < self.prob:
                rect1, rect2 = gen_nooverlap_rectangles(y, mask)
                if rect1[0] is not None:
                    x0_1, x0_2, y0_1, y0_2 = rect1
                    x1_1, x1_2, y1_1, y1_2 = rect2
                    y[i, :, x0_1:x0_2, y0_1:y0_2] = y[i, :, x1_1:x1_2, y1_1:y1_2].clone()
                    mask[i, :, x0_1:x0_2, y0_1:y0_2] = mask[i, :, x1_1:x1_2, y1_1:y1_2].clone()
        return y, mask

    def swap(self, y, mask_):
        """
        Copy-paste background and objects into other areas, without cutting semantic boundaries
        """
        ans = y.clone()
        mask = mask_.clone()
        for i in range(0, y.shape[0] - 1, 2):
            if random.random() < self.prob:
                for jj in range(5):
                    x1, x2, y1, y2 = gen_rectangle(y)
                    rect = x1, x2, y1, y2
                    if any_object_touched(rect, mask[i:i + 1]) or any_object_touched(rect, mask[i + 1:i + 2]):
                        continue
                    else:
                        ans[i, :, x1:x2, y1:y2] = y[i + 1, :, x1:x2, y1:y2].clone()
                        ans[i + 1, :, x1:x2, y1:y2] = y[i, :, x1:x2, y1:y2].clone()
                        mem = mask_[i, :, x1:x2, y1:y2].clone()
                        mask[i, :, x1:x2, y1:y2] = mask_[i + 1, :, x1:x2, y1:y2].clone()
                        mask[i + 1, :, x1:x2, y1:y2] = mem
                        break
            if random.random() < self.prob:
                which_object = torch.randint(mask.shape[1] - 1, size=()) + 1
                old_area = torch.argmax(mask[i], dim=0, keepdim=False) == which_object
                if not area_cut_any_object(old_area, mask[i + 1]):
                    ans[i+1] = ans[i].clone() * (old_area * 1.0) + ans[i+1].clone() * (1 - old_area * 1.0)
                    mask[i+1] = mask[i] * (old_area * 1.0) + mask[i+1] * (1 - old_area * 1.0)
        return ans, mask

    def move_objects(self, y, mask):
        """
        Move, dupplicate, or remove semantic objects
        """
        for i in range(0, y.shape[0]):
            num_changed_objects = torch.randint(mask.shape[1] - 1, size=()) + 1
            seq_classes = torch.randperm(mask.shape[1] - 1)[:num_changed_objects]
            for cur_class in seq_classes:
                old_area = torch.argmax(mask[i], dim=0, keepdim=False) == cur_class + 1  # +1 to avoid background
                new_area = generate_new_area(old_area, mask[i])
                if new_area[0] is None:
                    continue
                if random.random() < self.prob:
                    y[i], mask[i] = dupplicate_object(y[i], mask[i], old_area, new_area)
                if random.random() < self.prob:
                    y[i], mask[i] = remove_object(y[i], mask[i], old_area, new_area)
            return y, mask


# --- geometric helper functions --- #
def gen_rectangle(ans, w=-1, h=-1):
    x_c, y_c = random.random(), random.random()
    x_s, y_s = random.random()*0.4+0.1, random.random()*0.4+0.1
    x_l, x_r = x_c-x_s/2, x_c+x_s/2
    y_l, y_r = y_c-y_s/2, y_c+y_s/2
    x1, x2 = int(x_l*ans.shape[2]), int(x_r*ans.shape[2])
    y1, y2 = int(y_l*ans.shape[3]), int(y_r*ans.shape[3])
    if w < 0 or h < 0:
        pass
    else:
        x2, y2 = x1 + w, y1 + h
    x1, x2, y1, y2 = trim_rectangle(x1, x2, y1, y2, ans.shape)
    return x1, x2, y1, y2


def trim_rectangle(x1, x2, y1, y2, sh):
    if x1 < 0:
        x2 += (0 - x1)
        x1 += (0 - x1)
    if x2 >= sh[2]:
        x1 -= (x2 - sh[2] + 1)
        x2 -= (x2 - sh[2] + 1)
    if y1 < 0:
        y2 += (0 - y1)
        y1 += (0 - y1)
    if y2 >= sh[3]:
        y1 -= (y2 - sh[3] + 1)
        y2 -= (y2 - sh[3] + 1)
    return x1, x2, y1, y2


def gen_nooverlap_rectangles(ans, mask):
    x0_1, x0_2, y0_1, y0_2 = gen_rectangle(ans)
    for i in range(5):
        x1_1, x1_2, y1_1, y1_2 = gen_rectangle(ans, w=x0_2-x0_1, h=y0_2-y0_1)
        if not (x0_1 < x1_2 and x0_2 > x1_1 and y0_1 < y1_2 and y0_2 > y1_1):
            rect1, rect2 = [x0_1, x0_2, y0_1, y0_2], [x1_1, x1_2, y1_1, y1_2]
            if not any_object_touched(rect1, mask[i:i + 1]) and not any_object_touched(rect2, mask[i:i + 1]):
                return [x0_1, x0_2, y0_1, y0_2], [x1_1, x1_2, y1_1, y1_2]
    return [None, None, None, None], [None, None, None, None]  # if not found a good pair


def any_object_touched(rect, mask_):
    epsilon = 0.01
    x1, x2, y1, y2 = rect
    mask = torch.zeros_like(mask_)
    mask[:, 0, :, :] = mask_[:, 0, :, :]
    mask[:, 1:2, :, :] = torch.sum(torch.abs(mask_[:, 1:, :, :]), dim=1, keepdim=True)
    sum = torch.sum(mask[:, 1, x1:x2, y1:y2])
    if sum > epsilon:
        return True
    return False


def area_cut_any_object(area, mask_):
    epsilon = 0.01
    mask = torch.zeros_like(mask_)
    mask[0, :, :] = mask_[0, :, :]
    mask[1:2, :, :] = torch.sum(torch.abs(mask_[1:, :, :]), dim=0, keepdim=True)
    sum = torch.sum(area * mask[1, :, :])
    if sum > epsilon:
        return True
    return False


def generate_new_area(old_area, mask):
    epsilon = 0.01
    arg_mask = torch.argmax(mask, dim=0)
    if torch.sum(old_area) == 0:
        return None, None, None, None, None, None
    idx_x1 = torch.nonzero(old_area * 1.0)[:, 0].min()
    idx_x2 = torch.nonzero(old_area * 1.0)[:, 0].max()
    idx_y1 = torch.nonzero(old_area * 1.0)[:, 1].min()
    idx_y2 = torch.nonzero(old_area * 1.0)[:, 1].max()
    for i in range(5):
        new_x1 = torch.randint(0, mask.shape[1] - (idx_x2 - idx_x1), size=())
        new_y1 = torch.randint(0, mask.shape[2] - (idx_y2 - idx_y1), size=())
        x_diff = new_x1 - idx_x1
        y_diff = new_y1 - idx_y1
        provisional_area = torch.zeros_like(old_area)
        provisional_area[idx_x1+x_diff:idx_x2+x_diff+1, idx_y1+y_diff:idx_y2+y_diff+1] \
            = old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1]
        check_sum = torch.sum((provisional_area * 1.0) * arg_mask)
        if check_sum < epsilon:
            return x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2
    return None, None, None, None, None, None


def dupplicate_object(y, mask, old_area, new_area):
    x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2 = new_area

    y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] = \
        y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] \
        * (1.0 - old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * (1.0)) \
        + y[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        * (old_area[idx_x1:idx_x2 + 1,idx_y1:idx_y2 + 1] * 1.0)

    mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] = \
        mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] \
        * (1.0 - old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * (1.0)) \
        + mask[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        * (old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * 1.0)
    return y, mask


def remove_object(y, mask, old_area, new_area):
    x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2 = new_area

    y[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] = \
        y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1]

    mask[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        = mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1]
    return y, mask
