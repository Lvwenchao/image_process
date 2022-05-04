# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/5/3 14:49
# @FileName : utils.py
# @Software : PyCharm
import matplotlib.pyplot as plt


def show_img(imgs, titles):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[0])
    plt.title(titles[0])
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(imgs[1])
    plt.title(titles[1])
    plt.xticks([]), plt.yticks([])
    plt.show()
