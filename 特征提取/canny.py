# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/3 10:50
# @FileName : canny.py
# @Software : PyCharm
import cv2
from utils import gaussian_filter
import numpy as np


def canny(img, sigma=1.3, K=3, max_threshold=0.15, min_threshold=0.05):
    H, W = img.shape[:2]
    # 高斯滤波
    img_gauss = gaussian_filter(img, sigma=sigma, K=K)
    # 计算 x方向的图像梯度
    # cv2.cv_64F 避免负值为0
    sobel_x = cv2.Sobel(img_gauss, ddepth=cv2.CV_64F, dx=1, dy=0)
    # 取绝对值转unit型
    sobel_x = cv2.convertScaleAbs(sobel_x)
    # y方向梯度值
    sobel_y = cv2.Sobel(img_gauss, ddepth=cv2.CV_64F, dx=0, dy=1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    # 计算总的梯度值 sqrt(x2+y2) 或者 使用 hypot
    G = np.hypot(sobel_y, sobel_x)
    # 将范围限制到[0,255]
    G = G / G.max() * 255
    # 计算梯度方向
    theta = np.arctan2(sobel_y, sobel_x)
    tan_theta = np.tan(theta)
    # 计算角度值
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    # 非极大值抑制
    out = np.zeros(shape=(H, W))
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            try:
                p1 = 255.
                p2 = 255.
                if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] < 180):
                    p1 = G[y, x - 1]
                    p2 = G[y, x + 1]
                elif 22.5 <= angle[y, x] < 67.5:
                    p1 = G[y - 1, x + 1]
                    p2 = G[y + 1, x - 1]
                elif 57.5 <= angle[y, x] < 112.5:
                    p1 = G[y - 1, x]
                    p2 = G[y + 1, x]
                elif 112.5 <= angle[y, x] < 157.5:
                    p1 = G[y - 1, x - 1]
                    p2 = G[y + 1, x + 1]
                if G[y, x] >= p1 and G[y, x] >= p2:
                    out[y, x] = G[y, x]

            except Exception as e:
                print(e)
    # 双边阈值
    max_threshold = np.max(out) * max_threshold
    min_threshold = max_threshold * min_threshold

    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(out >= max_threshold)
    weak_i, weak_j = np.where((out <= max_threshold) & (out >= min_threshold))

    out[strong_i, strong_j] = strong
    out[weak_i, weak_j] = weak

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if out[i, j] == weak:
                try:
                    if (out[i + 1, j - 1] == strong or out[i + 1, j] == strong or out[i + 1, j + 1] == strong
                            or (out[i, j - 1] == strong) or (out[i, j + 1] == strong)
                            or (out[i - 1, j - 1] == strong) or (out[i - 1, j] == strong) or (
                                    out[i - 1, j + 1] == strong)):
                        out[i, j] = strong
                    else:
                        out[i, j] = 0
                except IndexError as e:
                    pass

    return out
