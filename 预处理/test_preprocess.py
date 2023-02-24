# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/2/23 15:08
# @FileName : test_preprocess.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt

from utils import show_image
from 预处理.图像增强 import *

src_image = cv2.imread("../sample_data/DSC00315.JPG")
src = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)


class TestAugment:
    def test_aug_random(self):
        for i in range(10):
            data_aug1 = transforms(image=src.copy())['image']
            show_image((1, 2, 1), src, "src")
            show_image((1, 2, 2), data_aug1, "brightness_aug")
            plt.show()
