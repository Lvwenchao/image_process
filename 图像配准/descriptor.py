# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/1/10 15:46
# @FileName : descriptor.py
# @Software : PyCharm
import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils import show_image


class HOG(object):
    def __init__(self, cell_size, bin_size, strides=1, block_size=2):
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.strides = strides
        self.block_size = block_size
        self.angle_unit = 360 // bin_size

    def cell_histogram(self, cell_mag, cell_angle):
        """
        单元梯度直方图
        :return:
        """
        cell_h, cell_w = cell_mag.shape
        cell_bins = [0] * self.bin_size
        for i in range(cell_h):
            for j in range(cell_w):
                min_bin, max_bin, mod = self.close_bin(cell_angle[i][j])
                # 加权分配
                cell_bins[min_bin] += (cell_mag[i][j] * (1 - (mod / self.angle_unit)))
                cell_bins[max_bin] += (cell_mag[i][j] * mod / self.angle_unit)
        return np.asarray(cell_bins, np.float32)

    def close_bin(self, angle):
        """
        cal bin of pixel angle in cell
        :return:
        """
        angle_unit = self.angle_unit
        bin_size = self.bin_size
        bin_idx = int(angle / self.angle_unit)
        # 便于后续使用双线性插值,计算中间的角度值 0-angle_uint 之间
        mod = angle % angle_unit
        if bin_idx == bin_size:
            return bin_idx - 1, bin_idx % bin_size, mod
        return bin_idx, (bin_idx + 1) % bin_size, mod

    def render_gradient(self, src_h, src_w, cell_gradient):
        """
        梯度显示
        :param src_h:
        :param src_w
        :param cell_gradient:
        :return:
        """
        hog_image = np.zeros((src_h, src_w), dtype=np.float32)
        cell_width = self.cell_size / 2
        max_mag = np.max(np.array(cell_gradient, np.float32))
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    # 原点位置
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return hog_image

    def compute_hog(self, data: np.ndarray):
        """
        定向梯度直方图
        :param data:
        :return:
        """
        # The gradient of x and y direction
        g_x = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5)
        g_y = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5)
        # magnitude and angle
        g_mag = np.sqrt(np.square(g_x) + np.square(g_y))
        g_ang = cv2.phase(g_x, g_y, angleInDegrees=True)
        # hog of cell
        height, width = data.shape[:2]
        cell_gradient_vec = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for cell_i in range(cell_gradient_vec.shape[0]):
            for cell_j in range(cell_gradient_vec.shape[1]):
                # mag and angle for every cell
                cell_mag = g_mag[cell_i * self.cell_size:(cell_i + 1) * self.cell_size,
                           cell_j * self.cell_size:(cell_j + 1) * self.cell_size]
                cell_ang = g_ang[cell_i * self.cell_size:(cell_i + 1) * self.cell_size,
                           cell_j * self.cell_size:(cell_j + 1) * self.cell_size]
                # 计算单元直方图
                cell_gradient_vec[cell_i][cell_j] = self.cell_histogram(cell_mag, cell_ang)

        # hog visit
        hog_image = self.render_gradient(height, width, cell_gradient_vec)

        # hog victor
        hog_vector = []
        for i in range(0, cell_gradient_vec.shape[0] - self.block_size + 1, self.strides):
            for j in range(0, cell_gradient_vec.shape[1] - self.block_size + 1, self.strides):
                # four cell cos one block
                block = np.hstack([cell_gradient_vec[i][j], cell_gradient_vec[i][j + 1],
                                   cell_gradient_vec[i + 1][j], cell_gradient_vec[i][j + 1]])
                # 计算block的标准差
                cell_grad_std = np.linalg.norm(block)
                if cell_grad_std != 0:
                    block = block / cell_grad_std
                hog_vector.extend(block)
        return np.asarray(hog_vector, np.float32), hog_image


class CFOG(object):
    def __init__(self, bin_size):
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


if __name__ == '__main__':
    img = cv2.imread("../sample_data/hog_feature_sample.png", cv2.IMREAD_GRAYSCALE)
    hog_agent = HOG(cell_size=8, bin_size=8, strides=4, block_size=2)
    vector, image = hog_agent.compute_hog(img)
    show_image((1, 1, 1), image, "hog image")
    plt.show()
    plt.close()
    pass
