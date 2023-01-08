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


def correlation_match(src_image, dst_image):
    pass


def distance_match(src_image, dst_image):
    """
    基于误差匹配
    :param src_image:
    :param dst_image:
    :return:
    """
    if len(src_image.shape) > 2:
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    if len(dst_image.shape) > 2:
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    src_h, src_w = src_image.shape
    dst_h, dst_w = dst_image.shape
    # 最小处所在的ssd
    min_value = sys.maxsize
    # min ssd location
    min_x, min_y = dst_w, dst_h
    match_value = np.zeros((dst_h - src_h + 1, dst_w - src_w + 1), np.float32)
    for i in range(0, dst_h - src_h + 1):
        for j in range(0, dst_w - src_w + 1):
            dst_patch = dst_image[i:i + src_h, j:j + src_w]
            temp = np.sum(np.square(dst_patch - src_image))
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


def ssd(template, dst):
    """
    误差平方和算法
    :param template:
    :param dst:
    :return:
    """
    return np.sum(np.square(dst - template))


def mad(template, dst):
    """
    平均绝对误差
    :param template:
    :param dst:
    :return:
    """
    return np.mean(np.abs(template - dst))


def sad(template, dst):
    """

    绝对误差和
    :param template:
    :param dst:
    :return:
    """
    return np.sum(np.abs(template - dst))


def ncc(src, src_mean, dst, dst_mean):
    src_sub_mean = src - src_mean
    dst_sub_mean = dst - dst_mean
    ncc1 = np.sum(src_sub_mean * dst_sub_mean)
    ncc2 = np.sum(np.power(src_sub_mean, 2))
    ncc3 = np.sum(np.power(dst_sub_mean, 2))
    ncc = ncc1 / np.sqrt(ncc2 * ncc3)
    return ncc


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
