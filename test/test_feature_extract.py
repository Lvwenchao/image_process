# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/4 9:42
# @FileName : test_feature_extract.py
# @Software : PyCharm
from unittest import TestCase
import cv2
import matplotlib.pyplot as plt

from 特征提取 import canny
from utils import *


class TestFeatureExtractCase(TestCase):
    file_path = "../sample_data/uav/DSC00315.JPG"
    img_bgr = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def test_canny(self):
        out = canny.canny(self.img_gray)
        show_image((1, 2, 1), self.img_rgb)
        show_image((1, 2, 2), out)
        plt.show()
