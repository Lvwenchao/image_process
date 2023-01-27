# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/1/10 15:46
# @FileName : descriptor.py
# @Software : PyCharm
import math

import numpy as np
import cv2

# Global variables #
float_tolerance = 1e-7


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
    def __init__(self, bin_size=9, k_size=3):
        self.bin_size = bin_size
        self.angle_uint = 180 // self.bin_size
        self.ksize = k_size
        self.sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        self.z_kernel = (1, 2, 1)

    def generetor_pixel_feature(self, image):
        h, w = image.shape
        g_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        g_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        pixel_feature = np.zeros([h, w, self.bin_size])
        for i in range(self.bin_size):
            pixel_feature[:, :, i] = np.abs(np.cos(self.angle_uint * i) * g_y + np.sin(self.angle_uint * i) * g_x)
        pixel_feature[pixel_feature < 0] = 0
        pixel_feature = self.gauss_filters(pixel_feature, z_cov=True)
        return pixel_feature

    def z_direc_conv(self, image, kernel=(1, 2, 1)):
        h, w, c = image.shape
        image_pad = np.pad(image, ((0, 0), (0, 0), (len(kernel) // 2, len(kernel) // 2)), mode="edge")
        re = np.zeros_like(image)

        for i in range(0, c):
            re[:, :, i] = np.sum(image_pad[:, :, i:i + len(kernel)] * kernel, axis=-1)
        return re

    def gauss_filters(self, image, z_cov=True):
        """
        gauss conv filters in x and y direction
        [1,2,1] filter in channel direction
        在通道和空间上进行平滑处理
        :return:
        """
        # k=3,sigma=0.8 k=5,sigma=1.1

        re = cv2.GaussianBlur(src=image, ksize=(self.ksize, self.ksize), sigmaX=self.sigma, sigmaY=self.sigma)
        if z_cov:
            re = self.z_direc_conv(re, kernel=self.z_kernel)
        return re


class SIFT(object):
    def __init__(self, sigma0=1.6, intvl=3, border_width=1, up_sample=1, kps_location_fit_num=5, eigenvalue_ratio=10):
        """

        :param sigma0: 基准层 sigma
        :param intvl: 组内层数
        :param up_sample: 是否在原图上进行上采样
        :param kps_location_fit_num: 关键点精确位置循环次数
        :param eigenvalue_ratio: 边缘响应阈值
        """
        self.octave_n = None
        self.sigma0 = sigma0
        self.intvl = intvl
        self.num_image_per_octave = self.intvl + 3
        self.border_width = border_width
        self.up_sample = up_sample
        self.kps_location_fit_num = kps_location_fit_num
        self.eigenvalue_ratio = eigenvalue_ratio

    def gen_base_image(self, image, assumed_blur=0.5):
        """
        为了避丢失最高层的分辨率，
        先将原始影像的尺度扩大一倍来生成第-1组图像，再做高斯模糊

        :param: image 输入影像
        :return:
        """
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # 假设已经对初始影像进行了 σ = 0.5 的高斯模糊
        sigma_diff = np.sqrt(self.sigma0 ** 2 - assumed_blur ** 2)
        base_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
        return base_image

    def cmp_octave_num(self, image):
        """
        计算金字塔的组数
        :return:
        """
        octave_n = int(round(np.log2(min(image.shape)) - 3)) + self.up_sample
        self.octave_n = octave_n

    def gen_gaussian_pyramid(self, image):
        """
        生成不同尺度的高斯影像
        :param image: 基础影像
        :return:
        """
        k = 2 ** (1 / self.intvl)
        gaussian_sigmas = [self.sigma0]
        for i in range(1, self.num_image_per_octave):
            pre_sigma = k ** (i - 1) * self.sigma0
            after_sigma = k * pre_sigma
            sigma_temp = np.sqrt(after_sigma ** 2 - pre_sigma ** 2)
            gaussian_sigmas.append(sigma_temp)
        # 构建高斯金字塔
        gaussain_images = []
        for i in range(self.octave_n):
            gaussian_images_octave = [image]
            # 不同尺度的高斯模糊
            for sigma in gaussian_sigmas[1:]:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                gaussian_images_octave.append(image)
            # 下一层，先降采样
            image = gaussian_images_octave[-3]
            image = cv2.resize(image, (image.shape[0] // 2, image.shape[1] // 2), interpolation=cv2.INTER_LINEAR)
            gaussain_images.append(np.asarray(gaussian_images_octave, np.float32))
        return gaussain_images

    def gen_dog_image(self, gaussian_images):
        """
        Difference pyramid
        生成高斯差分影像，每组的高斯影像层数为 S+3, 差分影像为 S+2
        :param gaussian_images: 高斯金字塔
        :return:
        """
        dog_images = []
        for gaussian_images_octave in gaussian_images:
            dog_images_octave = []
            # 相邻两层影像相减
            for first_image, second_image in zip(gaussian_images_octave, gaussian_images_octave[1:]):
                dog_images_octave.append(np.subtract(second_image, first_image))
            dog_images.append(np.asarray(dog_images_octave, np.float32))
        return dog_images

    def is_extreme_pts(self, sub_image, contrast_threshold):
        """
        :param sub_image 中心点所在的领域
        :param contrast_threshold 对比度阈值
        尺度空间极值点检测
        :return:
        """
        center_value = sub_image[1, 1, 1]
        if np.abs(center_value) > contrast_threshold:
            if center_value > 0:
                return np.max(sub_image) == center_value
            elif center_value < 0:
                return np.min(sub_image) == center_value
        return False

    def center_gradient(self, pixel_array):
        """
        计算中心点在xyz方向上的梯度
        :param pixel_array:
        :return:
        """
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.asarray([dx, dy, ds])

    def center_hessian(self, pixel_array):
        """
        计算 海塞矩阵
        :return:
        """
        center_pixel = pixel_array[1, 1, 1]
        # 计算三个方向上的二阶导
        # df/dxdx= f(x+1)-2f(x)+f(x-1)
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel + pixel_array[0, 1, 1]
        # dxy=f(x+1,y+1,σ)+ f(x-1,y-1,σ)-f(x+1,y-1,σ)-f(x-1,y+1)
        dxy = 0.25 * (pixel_array[1, 2, 2] + pixel_array[1, 0, 0] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2])
        dxs = 0.25 * (pixel_array[2, 1, 2] + pixel_array[0, 1, 0] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2])
        dys = 0.25 * (pixel_array[2, 2, 1] + pixel_array[0, 0, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1])
        return np.asarray([[dxx, dxy, dxs],
                           [dxy, dyy, dys],
                           [dxs, dys, dss]])

    def kps_location(self, center_y, center_x, octave_index, s_index, contrast_threshold,
                     dog_image_octave: np.ndarray):
        """
        关键点定位
        确定关键点的亚像素位置
        通过围绕每个极值的邻居进行二次拟合，对尺度空间极值的像素位置进行迭代细化
        :param octave_index: 金字塔组索引，便于计算原点位置
        :param contrast_threshold:低对比度阈值
        :param dog_image_octave: 中心点所在的 pixel cube
        :param center_x: 关键点中心点x
        :param center_y: 关键点中心点y
        :param s_index: 中心点所在层索引
        :return:
        """
        # loww 中进行五次循环找到关键点位置
        dog_h, dog_w = dog_image_octave.shape[1], dog_image_octave.shape[2]
        for attempt_i in range(self.kps_location_fit_num):
            # 极值点所在的图像块
            dog_image_octave = dog_image_octave[s_index - 1:s_index + 2,
                               center_y - 1:center_y + 2,
                               center_x - 1:center_x + 2]
            dog_image_octave = dog_image_octave.astype(np.float32) / 255.
            center_gradient = self.center_gradient(dog_image_octave)
            center_hessian = self.center_hessian(dog_image_octave)
            # 最小二乘法求线性解，得到回归系数
            delta_xyz = - np.linalg.lstsq(center_hessian, center_gradient, rcond=None)[0]
            # 如果偏移量小于0.5, 则不需要调整位置，跳出循环
            if (delta_xyz <= 0.5).all():
                # 去除低对比度
                # 重新计算响应值
                contrast_vlaue = dog_image_octave[1, 1, 1] - 0.5 * np.dot(center_gradient, delta_xyz)
                # ? 为什么要乘以 intvl
                if abs(contrast_vlaue) * self.intvl > contrast_threshold:
                    # 去除边缘响应值
                    # 计算hessian矩阵的秩和值
                    xy_hessian = center_hessian[:2, :2]
                    xy_hessian_trace = np.trace(xy_hessian)
                    xy_hessian_det = np.linalg.det(xy_hessian)
                    if xy_hessian_det > 0 and self.eigenvalue_ratio * xy_hessian_trace ** 2 < (
                            (self.eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                        kps = cv2.KeyPoint()
                        # 输入图像为上采样后的位置，octave索引从零开始，索引不需要减1
                        kps.pt = (
                            (center_x + delta_xyz[0]) * 2 ** octave_index,
                            (center_y + delta_xyz[1]) * 2 ** octave_index)
                        # 金字塔层数? 为什么要这样计算
                        kps.octave = octave_index
                        # 计算图像的尺度, octave_index+1 是因为输入的图像为原始图像的两倍
                        kps.size = self.sigma0 * (
                                2 ** ((octave_index + 1) + (s_index + delta_xyz[2]) / np.float32(self.intvl)))
                        kps.response = abs(contrast_vlaue)
                        return kps, s_index
            # 否则需要对位置进行更新，重新计算偏移值
            center_x += delta_xyz[0]
            center_y += delta_xyz[1]
            s_index += int(round(delta_xyz[2]))
            # 避免超过边界
            if center_y < self.border_width or center_y >= dog_h - self.border_width or center_x < self.border_width or center_x >= dog_w - self.border_width or s_index < 1 or s_index > self.intvl:
                print("纠正值超出边界")
                break
        return None

    def keypoints_detect_with_(self, dog_images, contrast_threshold=0.04, ):
        """
         关键点检测
        :return:
        """
        # 查找极值点
        # 对比度阈值
        contrast_threshold = np.floor(0.5 * contrast_threshold / self.intvl * 255)
        for o_index, dog_images_octave in enumerate(dog_images):
            for s_index in range(self.intvl):
                # (i,j) 为 3×3 数组的中心点
                for i in range(self.border_width, dog_images_octave.shape[1] - self.border_width):
                    for j in range(self.border_width, dog_images_octave.shape[2] - self.border_width):
                        sub_image = dog_images_octave[i:i + 3, i - 1:i + 2, j - 1:j + 2]
                        # 判断是否为极值点
                        if self.is_extreme_pts(sub_image, contrast_threshold):
                            # 极值点定位
                            kps, local_image_index = self.kps_location(i, j, o_index, s_index + 1, contrast_threshold,
                                                                       do)

        pass

    def compute_keypts_des(self, image):
        self.cmp_octave_num(image)
        if self.up_sample:
            base_image = self.gen_base_image(image)
        else:
            base_image = image
        gaussian_pyramid = self.gen_gaussian_pyramid(base_image)
        dog_images = self.gen_dog_image(gaussian_pyramid)
