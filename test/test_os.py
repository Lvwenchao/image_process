# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/6/27 14:24
# @FileName : test_os.py
# @Software : PyCharm
import os.path
from unittest import TestCase
from osgeo import gdal
import numpy as np


def write_image(filename, proj, trans, image_data):
    datatype = gdal.GDT_UInt16
    im_bands, im_width, im_height = image_data.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, im_height, im_width, im_bands, datatype)
    dataset.SetGeoTransform(trans)
    dataset.SetProjection(proj)
    for band in range(im_bands):
        dataset.GetRasterBand(band + 1).WriteArray(image_data[band])
    del dataset


class TestOs(TestCase):
    def test_walk(self):
        path = 'T50RMU_20220506T024551_B02_10m.jp2'
        filenamem, ext = os.path.splitext(path)
        print(filenamem[:-7] + "b432_10.tiff", ext)

    def test_data(self):
        ds = gdal.Open(r"F:\data\Sentinel-2\L2A\L2A_RGB_10m\T50RMU_20220506T024551_b432_10m.tiff")
        b4 = gdal.Open(
            r"F:\data\Sentinel-2\L2A\S2A_MSIL2A_20220506T024551_N9999_R132_T50RMU_20220623T101633.SAFE\GRANULE\L2A_T50RMU_A035879_20220506T025306\IMG_DATA\R10m\T50RMU_20220506T024551_B04_10m.jp2")
        b3 = gdal.Open(
            r"F:\data\Sentinel-2\L2A\S2A_MSIL2A_20220506T024551_N9999_R132_T50RMU_20220623T101633.SAFE\GRANULE\L2A_T50RMU_A035879_20220506T025306\IMG_DATA\R10m\T50RMU_20220506T024551_B03_10m.jp2")
        b2 = gdal.Open(
            r"F:\data\Sentinel-2\L2A\S2A_MSIL2A_20220506T024551_N9999_R132_T50RMU_20220623T101633.SAFE\GRANULE\L2A_T50RMU_A035879_20220506T025306\IMG_DATA\R10m\T50RMU_20220506T024551_B02_10m.jp2")
        Xsize = ds.RasterXSize
        Ysize = ds.RasterYSize
        bnum = ds.RasterCount
        data = ds.ReadAsArray(0, 0, Xsize, Ysize)
        data_4 = b4.ReadAsArray(0, 0, Xsize, Ysize)
        data_3 = b3.ReadAsArray(0, 0, Xsize, Ysize)
        data_2 = b2.ReadAsArray(0, 0, Xsize, Ysize)
        save_data = np.stack([data_4, data_3, data_2], axis=0)
        print(save_data.dtype)
        write_image(r"F:\data\Sentinel-2\L2A\L2A_RGB_10m\test.tiff", ds.GetProjection(), ds.GetGeoTransform(),
                    save_data)

    def test_cre(self):
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(r'F:\\data\\Sentinel-2\\L2A\\patch\\p_640s_640\\test.tiff', 320, 320, 3, gdal.GDT_Float32)
        print(ds)