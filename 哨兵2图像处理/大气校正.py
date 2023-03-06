# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/6/23 18:15
# @FileName : 大气校正.py
# @Software : PyCharm
import os
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = "F:/data/Sentinel-2/L1C"
    target_dir = "F:/data/Sentinel-2/L2A"
    dir_li = os.listdir(root_dir)
    for path in tqdm(dir_li):
        file_path = os.path.join(root_dir, path)
        os.system("L2A_Process --resolution 10 " + file_path + " --res_database_dir " + target_dir)
