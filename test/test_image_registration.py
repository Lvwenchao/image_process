# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2023/1/15 21:35
# @FileName : test_image_registration.py
# @Software : PyCharm
import cv2
import numpy as np

from 图像配准.image_match import Matcher
from utils import *
from 图像配准.descriptor import CFOG, SIFT


class TestRegistration:
    dst_img = cv2.imread("../sample_data/uav/DSC00315.JPG")
    temp_img = cv2.imread("../sample_data/uav/DSC00315_patch.png")
    dst_img_gray = convert_to_gray(dst_img)
    temp_img_gray = convert_to_gray(temp_img)

    def test_image_match(self):
        match_agent = Matcher()

        # match
        match_score, match_pts = match_agent.match(self.temp_img_gray, self.dst_img_gray, location_type="min")
        min_location = match_pts.reshape((1, 4, 2))
        show_location = cv2.polylines(self.dst_img.copy(), min_location, isClosed=True, color=(255, 0, 0), thickness=3,
                                      lineType=cv2.LINE_AA)
        ssd = cv2.normalize(match_score, match_score, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        show_image((3, 1, 1), show_location)
        show_image((3, 1, 2), self.temp_img)
        show_image((3, 1, 3), ssd)
        plt.show()

    def test_cfog(self):
        cfog_agent = CFOG()
        temp_pixel_feature = cfog_agent.generetor_pixel_feature(self.temp_img_gray)
        dst_pixel_feature = cfog_agent.generetor_pixel_feature(self.dst_img_gray)
        print(temp_pixel_feature.shape)
        ssd_match_agent = Matcher(method="ssd")
        match_value, match_pts = ssd_match_agent.match(temp_pixel_feature, dst_pixel_feature)


    def test_sift(self):
        sift_agent = SIFT()
        sift_agent.sift_cv(self.temp_img_gray)
