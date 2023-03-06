# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/11/20 16:00
# @FileName : HSV色彩检测.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 打开相机
capture = cv2.VideoCapture(0)

while 1:
    ret, frame = capture.read()

    # 转换到HSV空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV 蓝色阈值
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([150, 255, 255])

    # 根据阈值构建掩膜
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 按位运算
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow('mask', mask)
    cv2.imshow("res", res)
    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
