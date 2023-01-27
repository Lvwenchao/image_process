# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/12/19 21:33
# @FileName : ssd_match.py
# @Software : PyCharm
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import *


class Matcher(object):

    def __init__(self, method="ssd"):
        """
        基于灰度的图像///
        :param method:

        """
        self._function = {
            "ssd": self.ssd,
            "mad": self.mad,
            "sad": self.sad,
            "ncc": self.ncc,
            "mi": self.mutual_inf
        }
        self.method = self._function[method]

    def ncc(self, template, dst_image):
        tem_sub_mean = template - np.mean(template)
        dst_sub_mean = dst_image - np.mean(dst_image)
        ncc = np.sum(tem_sub_mean * dst_sub_mean) / (np.std(template) * np.std(dst_image))
        return ncc

    def match(self, template, dst_image, location_type="max"):
        """
        基于误差匹配
        :param self:
        :param template:
        :param dst_image:
        :param location_type: 根据相似度量和距离度量返回最大值或者最小值，max or min
        :return:
        """
        src_h, src_w = template.shape
        dst_h, dst_w = dst_image.shape
        # 最小处所在的ssd
        match_value = np.zeros((dst_h - src_h + 1, dst_w - src_w + 1), np.float32)
        for i in range(0, dst_h - src_h + 1):
            for j in range(0, dst_w - src_w + 1):
                dst_patch = dst_image[i:i + src_h, j:j + src_w]
                temp = self.method(template, dst_patch)
                match_value[i, j] = temp
        if location_type == "max":
            match_location = np.where(match_value == match_value.max())
        else:
            match_location = np.where(match_value == match_value.min())
        match_location = np.asarray(match_location).squeeze()
        y, x = match_location
        match_pts = np.asarray([[x, y], [x + src_w, y], [x + src_w, y + src_h], [x, y + src_h]])
        return match_value, match_pts

    def ssd(self, template, dst):
        """
        误差平方和算法
        :param template:
        :param dst:
        :return:
        """
        return np.sum(np.square(dst - template))

    def mad(self, template, dst):
        """
        平均绝对误差
        :param template:
        :param dst:
        :return:
        """
        return np.mean(np.abs(template - dst))

    def mutual_inf(self, template: np.ndarray, dst: np.ndarray):
        template = template.reshape(-1)
        dst = dst.reshape(-1)
        size = template.shape[-1]
        assert template.shape == dst.shape
        # 获取两个图像的直方图
        px = np.histogram(template, 256, (0, 255))[0] / size
        py = np.histogram(dst, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))
        hxy = np.histogram2d(template, dst, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))
        r = hx + hy - hxy
        return r

    def sad(self, template, dst):
        """

        绝对误差和
        :param template:
        :param dst:
        :return:
        """
        return np.sum(np.abs(template - dst))


