# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/3 14:49
# @FileName : utils.py
# @Software : PyCharm
import cv2
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


def conver_to_gray(image):
    assert len(image.shape) == 3
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
