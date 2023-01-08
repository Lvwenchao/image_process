# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/11/13 17:07
# @FileName : conver_2k.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image_path = r"G:\code\PyProject\image_process\sample_data\self.jpg"

img = cv2.imread(image_path)
img2 = cv2.resize(img, (45, 32),cv2.INTER_CUBIC)
cv2.namedWindow("self", 0)
cv2.imshow('self', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
