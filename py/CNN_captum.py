import numpy as np
import pandas as pd

import os.path as op
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from captum.attr import IntegratedGradients, GuidedGradCam
from matplotlib import pyplot as plt

torch.manual_seed(42)

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

year = 2017
images = np.memmap(op.join(r".\img_data\monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8,
                   mode='r').reshape(
    (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))

label_df = pd.read_feather(op.join(r".\img_data\monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
assert (len(label_df) == len(images))
label_df.head()


def imshow(img, vmin=None, vmax=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=vmin, vmax=vmax)
    plt.show(block=True)


if __name__ == '__main__':
    net = torch.load(
        r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_140817\categories of 2 ds_21_11_epoch_4_train_0.717940_val_0.691504.pt")
    net.eval()
    x = torch.Tensor(images[48].copy())  # can be set as arbitrary number
    x = x.reshape(-1, 1, 64, 60)
    x.requires_grad = True
    baseline = torch.zeros_like(x)
    ig = IntegratedGradients(net)
    attributions, delta = ig.attribute(x.cuda(), baseline.cuda(), target=0, return_convergence_delta=True)
    print('IG Attributions:', attributions)
    print('Convergence Delta:', delta)
    imshow(x[0].detach().cpu())
    imshow(abs(attributions.detach().cpu()[0]), vmin=0, vmax=0.02)
    imshow(attributions.detach().cpu()[0], vmin=0, vmax=0.02)
    imshow(-attributions.detach().cpu()[0], vmin=0, vmax=0.02)
    cam_layer_1 = GuidedGradCam(net, net.module.layer1)
    attributions_layer_1 = cam_layer_1.attribute(x.cuda(), target=1)
    cam_layer_2 = GuidedGradCam(net, net.module.layer2)
    attributions_layer_2 = cam_layer_2.attribute(x.cuda(), target=1)
    cam_layer_3 = GuidedGradCam(net, net.module.layer3)
    attributions_layer_3 = cam_layer_3.attribute(x.cuda(), target=1)
    imshow(abs(attributions_layer_1.detach().cpu()[0]), vmax=1e-8)
    imshow(abs(attributions_layer_2.detach().cpu()[0]), vmax=1e-7)
    imshow(abs(attributions_layer_3.detach().cpu()[0]), vmax=1e-7)
