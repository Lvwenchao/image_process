# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/6/23 18:16
# @FileName : 大气校正后处理.py
# @Software : PyCharm
import cv2
import glob
import numpy as np
import os
import tqdm
from osgeo import gdal

os.environ['PROJ_LIB'] = r"C:\Users\lwc\anaconda3\envs\py36\Lib\site-packages\osgeo\data\proj"


class Sentiel(object):
    def read_img(self, filename):
        """
        读取tiff影像
        :param filename:
        :return: proj, geotrans, data, Xsize, Ysize, bnum
        """
        ds = gdal.Open(filename)
        Xsize = ds.RasterXSize
        Ysize = ds.RasterYSize
        bnum = ds.RasterCount
        data = ds.ReadAsArray(0, 0, Xsize, Ysize)
        proj = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        return proj, geotrans, data, Xsize, Ysize, bnum

    def write_image(self, save_filename, proj, trans, image_data):
        """
        注意save_filename所在的文件夹需要先创建
        :param save_filename:
        :param proj:
        :param trans:
        :param image_data:
        :return:
        """
        if 'int8' in image_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in image_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        im_bands, im_width, im_height = image_data.shape

        driver = gdal.GetDriverByName('GTiff')
        # 如果为None 可能是地址的原因
        dataset = driver.Create(save_filename, im_height, im_width, im_bands, datatype)
        dataset.SetGeoTransform(trans)
        dataset.SetProjection(proj)
        for band in range(im_bands):
            dataset.GetRasterBand(band + 1).WriteArray(image_data[band])
        del dataset


def multi_band_fusion():
    """
    RGB波段融合
    :return:
    """
    # tiff影像保存地址
    s_agent = Sentiel()

    tiff_save_path = r"F:\data\Sentinel-2\L2A\tiff"
    # 循环遍历每个子文件夹
    l2c_dir = r"F:\data\Sentinel-2\L2A"
    # dir_list = [_ for _ in os.listdir(test_path) if "L2A" in _]
    # save_rgb_dir = r"F:\data\Sentinel-2\L2A\L2A_RGB_10"
    save_rgb_dir = r"F:\data\Sentinel-2\L2A\L2A_RGB_TEST"
    if not os.path.exists(save_rgb_dir):
        os.mkdir(save_rgb_dir)
    # # 遍历每个 L2A文件夹：
    b2, b3, b4 = None, None, None
    num = 1
    for root, dirs, files in os.walk(l2c_dir):
        for file in files:
            file_name = os.path.join(root, file)
            # 获
            if "B02_10m" in file_name:
                proj, geotrans, b2, Xsize, Ysize, bnum = s_agent.read_img(file_name)
            if "B03_10m" in file_name:
                proj, geotrans, b3, Xsize, Ysize, bnum = s_agent.read_img(file_name)
            if "B04_10m" in file_name:
                proj, geotrans, b4, Xsize, Ysize, bnum = s_agent.read_img(file_name)

            # 将三个波段的数据进行整合，并且将反射率归 0-1之间
            if b4 is not None and b3 is not None and b2 is not None:
                save_path = os.path.join(save_rgb_dir, os.path.splitext(file)[0][:-7] + "b432_10m.tiff")
                # [b,w,h]
                merge = np.stack([b4, b3, b2], axis=0)
                # 0-10000范围
                merge[merge > 10000] = 10000
                # 0-255
                # merge = merge * 0.0001 * 255
                # merge[merge > 255] = 255.0
                # merge = merge.astype(np.uint8)
                # 0-1
                # merge[merge>1]=1
                s_agent.write_image(save_path, proj, geotrans, merge)
                print("fusion {} result in {}".format(num, save_path))
                b2, b3, b4 = None, None, None
                num += 1


def image_clip(root_dir, patch_dir):
    patch_size = 640
    # 通过stride觉得相邻影像的重叠部分
    stride = patch_size
    # 裁剪文件保存路径
    patch_save_dir = os.path.join(patch_dir, 'p{}s{}'.format(patch_size, stride))
    if not os.path.exists(patch_save_dir):
        os.mkdir(patch_save_dir)
    s_agent = Sentiel()

    filenames = os.listdir(root_dir)
    filepaths = [os.path.join(root_dir, _) for _ in filenames]
    count = 0
    for index, path in enumerate(tqdm.tqdm(filepaths)):
        proj, geotrans, data, w, h, bands, = s_agent.read_img(path)
        # 补全周围像素
        column_num = (w - patch_size) // stride + 1
        row_num = (h - patch_size) // stride + 1
        column_num = column_num + 1
        row_num = row_num + 1
        colum_pad = (column_num * patch_size - w) // 2
        row_pad = (row_num * patch_size - h) // 2
        data = np.pad(data, ((0, 0), (row_pad, row_pad), (colum_pad, colum_pad)), mode='reflect')
        for i in range(1, row_num + 1):
            for j in range(1, column_num + 1):
                sample = data[:, stride * (i - 1):stride * (i - 1) + patch_size,
                         stride * (j - 1):stride * (j - 1) + patch_size]
                totals = patch_size * patch_size * bands
                zero_num = np.sum(sample == 0)
                if zero_num / totals > 0.4:
                    continue
                save_path = os.path.join(patch_save_dir, '{}{}'.format(str(count).zfill(6), '.tif'))
                s_agent.write_image(save_path, proj, geotrans, sample)
                count += 1
        if count // 1000 == 0:
            print("----{} images".format(count))


if __name__ == '__main__':
    multi_band_fusion()
    # root_dir = r"F:\data\Sentinel-2\L2A\L2A_RGB_10m"
    # patch_dir = r"F:\data\Sentinel-2\L2A\patch"
    # image_clip(root_dir, patch_dir)
