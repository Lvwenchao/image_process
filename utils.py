# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/3 14:49
# @FileName : utils.py
# @Software : PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(location, img, title=None, width=None, text=None, save_path=None, save_dict=None):
    if width is not None:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    if title is not None:
        plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if text is not None:
        plt.text(0.5, 0.25, text)
    if save_path is not None:
        plt.savefig(save_path,
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0)
    if width is not None:
        plt.show()
        plt.close()


def convert_to_gray(image):
    assert len(image.shape) == 3
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_filter(img, sigma=1.3, K=3):
    """

    :param img: 滤波图像
    :param sigma: 标准差
    :param K: 模板大小
    :return:
    """
    k = K // 2
    if len(img.shape) == 3:
        H, W, C = img.shape

    else:

        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    out = np.pad(img.copy(), ((k, k), (k, k), (0, 0)))

    # 计算滤波核
    kernel = np.zeros((K, K), np.float32)
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            kernel[y + k, x + k] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    kernel = kernel / (2.0 * np.pi * (sigma ** 2.0))

    # 当核为整数时进行归一化
    kernel = kernel / np.sum(kernel)

    # 进行卷积操作
    temp = out.copy()
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[y + k, x + k, c] = np.sum(kernel * temp[y:y + K, x:x + K, c])

    out = np.clip(out, 0, 255)

    out = out[k:H + k, k:W + k, :].astype(np.uint8)

    return out


def gaussian_filter_cv(img, sigma=1.3, K=3):
    out = cv2.GaussianBlur(img, (K, K), sigma)
    return out


def ums(image_path, w=0.6):
    src = cv2.imread(image_path)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    cv2.imshow("mask image", usm)

    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0:h, 0:w, :] = src
    result[0:h, w:2 * w, :] = usm
    # cv2.putText(result, "original image", (10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv2.putText(result, "sharpen image", (w + 10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv.putText(图像名，标题，（x坐标，y坐标），字体，字的大小，颜色，字的粗细）

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("../sample_data/filter_re.png", usm)
