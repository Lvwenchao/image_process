# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/10/10 9:09
# @FileName : k_means.py
# @Software : PyCharm
# k 均值分类
# 根据k个最近的邻居占比进行分类
import random

import cv2.ml
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)
# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
# Labels each one either Red or Blue with numbers 0 and 1
labels = np.random.randint(0, 2, (25, 1)).astype(np.float32)
# Take Red families and plot them
# ravel()方法将数组维度拉成一维数组
red = trainData[labels.ravel() == 0]
ax = plt.subplot(111)
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
# Take Blue families and plot them
blue = trainData[labels.ravel() == 1]
ax.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
# 测试样本
test_sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)
ax.scatter(test_sample[:, 0], test_sample[:, 1], 80, 'g', 'o')

# 建立KNN分类器
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)
ret, results, neighbours, dist = knn.findNearest(test_sample, 3)
print("result: ", results, "\n")
print("neighbours: ", neighbours, "\n")
print("distance: ", dist)

plt.show()
