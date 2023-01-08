# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/12/7 15:44
# @FileName : hog.py
# @Software : PyCharm
# HOG 特征描述符
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import show_image


def close_bin(angle, angle_unit, bin_size):
    """
    判断输入哪个bin
    :param angle:
    :param angle_unit:
    :param bin_size
    :return:
    """
    bin_idx = int(angle / angle_unit)
    # 便于后续使用双线性插值
    mod = angle % angle_unit

    if bin_idx == bin_size:
        return bin_idx - 1, bin_idx % bin_size, mod
    return bin_idx, (bin_idx + 1) % bin_size, mod


def cell_histogram(cell_mag, cell_angle, bin_size, angle_unit):
    """
    计算每个单元的直方图
    :param angle_unit:
    :param bin_size:
    :param cell_mag:
    :param cell_angle:
    :return:
    """
    cell_h, cell_w = cell_mag.shape[:2]
    cell_bins = [0] * bin_size
    for i in range(cell_h):
        for j in range(cell_w):
            gradient = cell_mag[i][j]
            angle = cell_angle[i][j]
            min_bin, max_bin, mod = close_bin(angle, angle_unit, bin_size)
            cell_bins[min_bin] += (gradient * (1 - (mod / angle_unit)))
            cell_bins[max_bin] += (gradient * mod / angle_unit)
    return np.asarray(cell_bins, np.float32)


def render_gradient(image, cell_gradient, cell_size, angle_unit):
    cell_width = cell_size / 2
    max_mag = np.max(np.array(cell_gradient, np.float32))
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    return image


def hog(data, cell_size, bin_size, strides=1, block_size=2):
    """
    计算图像的梯度直方图
    :return:
    """
    # 角单元
    angle_unit = 360 // bin_size

    # 计算方向梯度
    g_x = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5)
    g_y = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5)
    # 计算每个像素点的梯度幅值和方向
    g_mag = np.sqrt(np.square(g_x) + np.square(g_y))
    g_ang = cv2.phase(g_x, g_y, angleInDegrees=True)
    # 每个区域的直方图
    height, width = data.shape[:2]
    cell_vector = np.zeros((height // cell_size, width // cell_size, bin_size))
    for cell_i in range(cell_vector.shape[0]):
        for cell_j in range(cell_vector.shape[1]):
            # 提出每个单元
            cell_mag = g_mag[cell_i * cell_size:(cell_i + 1) * cell_size, cell_j * cell_size:(cell_j + 1) * cell_size]
            cell_ang = g_ang[cell_i * cell_size:(cell_i + 1) * cell_size, cell_j * cell_size:(cell_j + 1) * cell_size]
            # 计算单元直方图
            cell_vector[cell_i][cell_j] = cell_histogram(cell_mag, cell_ang, bin_size, angle_unit)

    # 每个block 由 四个cell 组成
    # 进行标准化
    hog_image = render_gradient(np.zeros([height, width]), cell_vector, cell_size, angle_unit)
    hog_vector = []
    for i in range(0, cell_vector.shape[0] - block_size + 1, strides):
        for j in range(0, cell_vector.shape[1] - block_size + 1, strides):
            block = np.hstack([cell_vector[i][j], cell_vector[i][j + 1], cell_vector[i + 1][j], cell_vector[i][j + 1]])
            # 计算block的标准差
            cell_grad_std = np.linalg.norm(block)
            if cell_grad_std != 0:
                block = block / cell_grad_std
            hog_vector.extend(block)
    return np.asarray(hog_vector, np.float32), hog_image


class CFOR(object):
    def __int__(self, bin_size):
        self.bin_size = bin_size
        self.angle_uint = 180 // self.bin_size
        self.z_kernel = (1, 2, 1)

    def generetor_pixel_feature(self, image):
        h, w = image.shape
        g_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        g_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        pixel_feature = np.zeros([h, w, self.bin_size])
        for i in range(self.bin_size):
            pixel_feature[:, :, i] = np.abs(np.cos(self.angle_uint * i) * g_y + np.sin(self.angle_uint * i) * g_x)
        pixel_feature[pixel_feature < 0] = 0
        return pixel_feature

    def z_direc_conv(self, image, kernel=(1, 2, 1)):
        h, w, c = image.shape
        kernel_size = kernel.shape[0]
        image = np.expand_dims(image)
        image_pad = np.pad(image, ((0, 0), (0, 0), (kernel_size // 2, kernel_size // 2)), mode="edge")
        re = np.zeros_like(image)
        for i in range(0, c):
            re[:, :, i] = image_pad[:, :, i:i + kernel_size] * kernel
        return re

    def gauss_filters(self, image, k_size, z_cov=True):
        """
        gauss conv filters in x and y direction
        [1,2,1] filter in channel direction
        在通道和空间上进行平滑处理
        :return:
        """
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        re = cv2.GaussianBlur(src=image, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma)
        if z_cov:
            re = self.z_direc_conv(re, kernel=self.z_kernel)
        return re

    def ssd_match(self, src, dst):
        
        pass


if __name__ == '__main__':
    # img = cv2.imread("../sample_data/hog_feature_sample.png", cv2.IMREAD_GRAYSCALE)
    # vector, image = hog(img, cell_size=8, bin_size=8, strides=4)
    # show_image((1, 1, 1), image, "hog image")
    # plt.show()
    # plt.close()
    pass
