# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/2/23 14:38
# @FileName : 图像增强.py
# @Software : PyCharm
from typing import Tuple, Dict, Any

import numpy as np
import cv2
import albumentations as a

noise_transform_list = [
    a.GaussNoise(),
    a.MultiplicativeNoise(),
    a.GaussianBlur(),
    a.MotionBlur(),
    a.MedianBlur(),
    a.ISONoise(),
]

transforms = a.Compose([
    # 噪声增强
    # a.OneOf(noise_transform_list, p=0.5),
    # 翻转增强
    a.HorizontalFlip(p=0.5),
    a.VerticalFlip(p=0.5),
    # 亮度增强
    a.RandomBrightnessContrast(p=0.5),
], p=0.6)
