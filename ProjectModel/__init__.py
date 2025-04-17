# -*- coding: UTF-8 -*-
"""
*@ description: 初始化package
*@ name:	__init__.py
*@ author: dengbozfan
*@ time:	2025/04/12 22:03
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
import csv
import json

import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm

import math
from functools import singledispatch

from .utils.ProcessImage import normalize,closest_power_of_two
from .utils.bulid_dataset import bulid_dataset
from .dataset.pointdataset import PointDataset
from .dataset.polygendataset import PolygonDataset

from .utils.Evaluator.baseEvaluator import (
    Accumulator,SegmentationMetric,
    evaluate_segmentation_iou,evaluate_segmentation_loss
)
from .utils.Evaluator.plot_result import PlotResult
from .utils.Evaluator.MalariaEvaluator import MalariaEvaluator


from .block.ResNet import resnet18
from .net.Unet_up_ResNext import UNet_up_Resnet18
from .use.train import Train

