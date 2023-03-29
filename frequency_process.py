# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/3/4 10:29
# @FileName : frequency_process.py
# @Software : PyCharm
"""
频率域处理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"D:\code\PyProject\image_process\sample_data\DSC00315.JPG", 0)
# 尺度数量
n_scales = 4
# 方向
n_angles = 6
H, W = image.shape[:2]

min_wavelength = 12
multiplier = 0.55


def lowpassfilter(H, W, cutoff, n):
    """
    低通滤波卷积核
    :param H:
    :param W:
    :param cutoff:
    :param n:
    :return:
    """
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('the cutoff frequency needs to be between 0 and 0.5')

    if not n == int(n) or n < 1.0:
        raise ValueError('n must be an integer >= 1')

    xrange = np.linspace(-0.5, 0.5, W)
    yrange = np.linspace(-0.5, 0.5, H)

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    radius = np.fft.ifftshift(radius)
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * n))


xrange = np.linspace(-0.5, 0.5, W)
yrange = np.linspace(-0.5, 0.5, H)
x, y = np.meshgrid(xrange, yrange)
radius = np.sqrt(x ** 2 + y ** 2)
theta = np.arctan2(-y, x)

# numpy.fft模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央。ifftshift函数则是其逆操作
radius = np.fft.ifftshift(radius)
theta = np.fft.ifftshift(theta)

sintheta = np.sin(theta)
costheta = np.cos(theta)

lp_filter = lowpassfilter(H, W, 0.45, 15)
# lp_filter = np.fft.ifftshift(lp_filter)
log_gabor_filters = np.zeros((n_scales, H, W))

#  不同尺度
for sc in range(n_scales):
    sigma_onf = sc / 2
    wavelength = min_wavelength * multiplier ** sc
    log_gabor_filters[sc] = np.exp(
        (-(np.log(radius * wavelength + 1e-5)) ** 2) / (2 * np.log(sigma_onf + 1e-5) ** 2)) * lp_filter

spreads = np.zeros((n_angles, H, W))
#  不同方向
for o in range(n_angles):
    angle = o * np.pi / n_angles
    ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
    dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
    dtheta = abs(np.arctan2(ds, dc))
    dtheta = np.minimum(dtheta * n_angles * 0.5, np.pi)
    spreads[o] = (np.cos(dtheta) + 1) / 2
#  构造集合的filter
filter_bank = np.zeros((n_scales * n_angles, H, W))
for sc in range(n_scales):
    for o in range(n_angles):
        filter_bank[sc * n_angles + o] = log_gabor_filters[sc] * spreads[o]

#  可视化


image_fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
eo = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2], 2))
for i, filter_i in enumerate(filter_bank):
    eo[i] = cv2.idft(np.multiply(np.expand_dims(filter_i, -1), image_fft))

for sc in range(n_scales):
    plt.figure(sc, figsize=(30, 120))
    first = eo[n_scales]
    first /= np.max(first)
    first *= 255
    shuiping = first
    for o in range(n_angles - 1):
        out = eo[sc * n_angles + o + 1] - np.min(eo[sc * n_angles + o + 1])
        out /= np.max(out)
        out *= 255
        shuiping = np.concatenate([shuiping, out], axis=1)
        plt.imshow(shuiping[:, :, 0], cmap="gray")
        plt.show()
