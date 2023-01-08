# -*- using coding=utf-8 -*-

# 设置尺寸大小为224*224
import glob
import os
import cv2
from gdalclip3 import GRID
import numpy as np

# item例子
"""
用于产生样本,样本的尺寸大小为224*224
可以使用这个函数产生训练集合、验证集的样本 (可以在里面添加样本增强的函数？？)
train_sample_path, label_sample_path 是产生样本、对应的标签的路径
label_data_list是原始标签的txt的位置，原始标签txt中，每行存储
label_path, train_path是原始标签、样本存放的路径
"""


def generate_sample(label_data_list, size=224, stride=112, label_path=None, train_path=None,
                    label_sample_path='/media/tom/file/potsdam/label_id_val/',
                    train_sample_path='/media/tom/file/potsdam/RGB_val/'):
    # with open(label_data_list, 'r') as txt:
    #    label_data_list=[i.strip('\n') for i in txt.readlines()]

    for item in label_data_list:
        sample_name = item.split('/')[-1].split('.')[0]  # 文件名

        label_data_path = os.path.join(label_path, '{}.tif'.format(sample_name))  # item  #
        train_data_path = os.path.join(train_path, '{}.tiff'.format(sample_name))

        label_sample_path = label_sample_path
        train_sample_path = train_sample_path

        label_proj, label_geotrans, label_data = GRID().read_img(label_data_path)
        train_proj, train_geotrans, train_data = GRID().read_img(train_data_path)

        label_data = label_data[0]
        h, w = label_data.shape

        # column_num = (w-size)//stride+1; row_num = (h-size)//stride+1

        # 补全周围像素的
        column_num = (w - size) // stride + 1;
        row_num = (h - size) // stride + 1
        column_num = column_num + 1;
        row_num = row_num + 1
        colum_pad = (column_num * size - w) // 2;
        row_pad = (row_num * size - h) // 2

        label_data = np.pad(label_data, ((row_pad, row_pad), (colum_pad, colum_pad)), mode='reflect')
        train_data = np.pad(train_data, ((0, 0), (row_pad, row_pad), (colum_pad, colum_pad)), mode='reflect')

        for i in range(1, row_num + 1):
            for j in range(1, column_num + 1):
                label_sample = label_data[stride * (i - 1):stride * (i - 1) + size,
                               stride * (j - 1):stride * (j - 1) + size]
                train_sample = train_data[:, stride * (i - 1):stride * (i - 1) + size,
                               stride * (j - 1):stride * (j - 1) + size]
                train_sample = train_sample.astype(np.int8)
                GRID().write_img(filename='{}{}_{}_{}{}'.format(label_sample_path, sample_name, i, j, '.tif'),
                                 im_proj=label_proj, im_geotrans=label_geotrans, im_data=label_sample)
                GRID().write_img(filename='{}{}_{}_{}{}'.format(train_sample_path, sample_name, i, j, '.tif'),
                                 im_proj=train_proj,
                                 im_geotrans=train_geotrans, im_data=train_sample)


if __name__ == '__main__':
    """
    massachusetts2 文件夹的尺寸是224*224, stride:0
    """
    file_type = '../test'
    label_data_list = glob.glob('/media/tom/file/massachusetts_tiff/{}/*.tiff'.format(file_type))

    generate_sample(label_data_list=label_data_list,
                    label_path='/media/tom/file/massachusetts_tiff/{}_labels'.format(file_type),
                    train_path='/media/tom/file/massachusetts_tiff/{}'.format(file_type),
                    label_sample_path='/media/tom/file/massachusetts2/{}_labels/'.format(file_type),
                    train_sample_path='/media/tom/file/massachusetts2/{}/'.format(file_type),
                    size=224,
                    stride=224)
