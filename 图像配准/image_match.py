# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/12/19 21:33
# @FileName : ssd_match.py
# @Software : PyCharm
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import show_image


class Matcher(object):

    def __init__(self, method="ssd", type="distance"):
        """
        基于灰度的图像///
        :param method:
        :param type:
        """
        self.method = method
        self._function = {
            "ssd": self.ssd,
            "mad": self.mad,
            "sad": self.sad,
            "ncc": self.method,
            "mi": self.mutual_inf
        }

    def ncc(self,):

    def match(self, template, dst_image):
        """
        基于误差匹配
        :param self:
        :param template:
        :param dst_image:
        :return:
        """
        if len(template.shape) > 2:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(dst_image.shape) > 2:
            dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
        src_h, src_w = template.shape
        dst_h, dst_w = dst_image.shape
        # 最小处所在的ssd
        min_value = sys.maxsize
        # min ssd location
        min_x, min_y = dst_w, dst_h
        match_value = np.zeros((dst_h - src_h + 1, dst_w - src_w + 1), np.float32)
        for i in range(0, dst_h - src_h + 1):
            for j in range(0, dst_w - src_w + 1):
                dst_patch = dst_image[i:i + src_h, j:j + src_w]
                temp = np.sum(np.square(dst_patch - template))
                match_value[i, j] = temp
                # 获取最小ssd
                if temp < min_value:
                    min_value = temp
                    min_x, min_y = j, i
        min_location = np.asarray([[min_x, min_y], [min_x + src_w, min_y],
                                   [min_x + src_w, min_y + src_w], [min_x, min_y + src_w]],
                                  dtype=np.int32)
        print("match_score:{}\npoints:{}".format(str(min_value), min_location))
        return match_value, min_value, min_location

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


if __name__ == '__main__':
    dst_image = cv2.imread("../sample_data/uav/DSC00315.JPG")
    src_image = cv2.imread("../sample_data/uav/DSC00315_patch.png")
    match_score, best_score, match_pts = distance_match(src_image, dst_image)
    min_location = match_pts.reshape((1, 4, 2))
    show_location = cv2.polylines(dst_image.copy(), min_location, isClosed=True, color=(255, 0, 0))
    ssd = cv2.normalize(match_score, match_score, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    show_image((3, 1, 1), show_location)
    show_image((3, 1, 2), src_image)
    show_image((3, 1, 3), ssd)
    plt.show()
