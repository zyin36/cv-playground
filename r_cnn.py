import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt


def get_iou(bb1, bb2):
  # assuring for proper dimension.
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
  # calculating dimension of common area between these two boxes.
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
  # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
  # calculating intersection area.
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
  # individual areas of both these bounding boxes.
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
  # union area = area of bb1_+ area of bb2 - intersection of bb1 and bb2.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# Ref: https://arxiv.org/pdf/1311.2524
class BBox_Regressor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        ## Regression layers
        #  L2 lambda = 1000
        self.reg_param_x = nn.Linear(feature_dim, 1, bias=False)
        self.reg_param_y = nn.Linear(feature_dim, 1, bias=False)
        self.reg_param_w = nn.Linear(feature_dim, 1, bias=False)
        self.reg_param_h = nn.Linear(feature_dim, 1, bias=False)

        def forward(self, p, features):
            # shape: (batch_dim)
            p_x, p_y, p_w, p,h = p[:,0], p[:,1], p[:,2], p[:,3]
            g_x = p_w * reg_param_x(features) + p_x
            g_y = p_h * reg_param_y(features) + p_y
            g_w = p_w * torch.exp(reg_param_w(features))
            g_h = p_h * torch.exp(reg_param_h(features))
            return torch.hstack(g_x, g_y, g_w, g_h)
            