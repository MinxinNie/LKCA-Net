
import time
from random import random

from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from functools import partial
from multiprocessing import Pool
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
import logging
import logging.handlers

from segmentation_mask_overlay import overlay_masks


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'ISIC2018':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'ISIC2017':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_metric_list_percase(pred, gt, classes=9):
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred == i, gt == i))
    return np.array(metric_list)

def calculate_metric_multicases(preds, gts, classes=9, num_workers=12):
    with Pool(num_workers) as p:
        metrics_list = p.starmap(partial(calculate_metric_list_percase, classes=classes), zip(preds, gts))
    metrics_list = np.array(metrics_list)
    metrics_list = metrics_list.mean(axis=0) # 8x2
    return metrics_list

# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,
                       class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names == None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink',
                 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes - 1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                # print(len(P))
                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out

                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl == i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred == i)
                prediction[ind] = pred

                if test_save_path is not None:
                    fig_gt = overlay_masks(image[ind, :, :], np.stack(masks, -1), labels=mask_labels, colors=cmap,
                                           alpha=0.5, return_type='mpl')
                    fig_pred = overlay_masks(image[ind, :, :], np.stack(preds_o, -1), labels=mask_labels, colors=cmap,
                                             alpha=0.5, return_type='mpl')
                    # Do with that image whatever you want to do.
                    fig_gt.savefig(test_save_path + '/' + case + '_' + str(ind) + '_gt.png', bbox_inches="tight",
                                   dpi=300)
                    fig_pred.savefig(test_save_path + '/' + case + '_' + str(ind) + '_pred.png', bbox_inches="tight",
                                     dpi=300)
                    plt.close('all')
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    return metric_list



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def make_dirs_by_time(save_dir):
    version = str(time.time())
    save_dir = join(save_dir, f"exp_{version}")
    if not os.path.exists(save_dir):
        maybe_mkdir_p(save_dir)

    return save_dir

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


join = os.path.join