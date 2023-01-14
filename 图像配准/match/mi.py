# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/12/22 15:20
# @FileName : MI.py
# @Software : PyCharm
import sys

import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import mutual_info_score
from utils import conver_to_gray, show_image


def mutual_information_match(template, dst_image):
    """
    互信息匹配

    :param template: 模板
    :param dst_image:
    :return:
    """
    # 先获取两幅图像的灰度直方图
    # mi = MI()
    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if len(dst_image.shape) > 2:
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    src_h, src_w = template.shape
    dst_h, dst_w = dst_image.shape
    # 最小处所在的ssd
    max_match_value = -255
    # min ssd location
    max_x, max_y = dst_w, dst_h
    mi = np.zeros((dst_h - src_h + 1, dst_w - src_w + 1), np.float32)
    for i in tqdm.tqdm(range(0, dst_h - src_h + 1)):
        for j in range(0, dst_w - src_w + 1):
            dst_patch = dst_image[i:i + src_h, j:j + src_w]
            temp = mutual_inf(dst_patch, template)
            mi[i, j] = temp
            # 获取最小ssd
            if temp > max_match_value:
                max_match_value = temp
                max_x, max_y = j, i
    match_location = np.asarray([[max_x, max_y], [max_x + src_w, max_y],
                                 [max_x + src_w, max_y + src_w], [max_x, max_y + src_w]],
                                dtype=np.int32)

    return mi, max_match_value, match_location





if __name__ == '__main__':
    template = cv2.imread("../sample_data/uav/DSC00315_patch.png")
    template_gray = conver_to_gray(template)
    dst = cv2.imread("../sample_data/uav/DSC00315.JPG")
    dst_gray = conver_to_gray(dst)
    value, max_value, match_location = mutual_information_match(template_gray, dst_gray)
    match_location = match_location.reshape((1, 4, 2))
    show_location = cv2.polylines(dst.copy(), match_location, isClosed=True, color=(0, 255, 0))
    show_image((3, 1, 1), show_location)
    show_image((3, 1, 2), template)
    show_image((3, 1, 3), value)
    plt.show()
