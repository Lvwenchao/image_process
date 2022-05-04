# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/3 14:35
# @FileName : test_filter.py
# @Software : PyCharm
from unittest import TestCase
from src.filter import gaussian_filter
from src.utils import show_img
import cv2


class TestFilterCase(TestCase):
    file_path = "../sample_data/DSC00315.JPG"
    img_bgr = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def test_gauss_filter(self):
        out = gaussian_filter(self.img_rgb)
        show_img([self.img_rgb, out], ['filter before', 'filter after'])
