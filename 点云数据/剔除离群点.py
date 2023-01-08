# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/10/18 20:55
# @FileName : 剔除离群点.py
# @Software : PyCharm
import open3d as o3d

source_pts = o3d.io.read_point_cloud(filename="../sample_data/plys/6.ply")  # source 为需要配准的点云
target_pts = o3d.io.read_point_cloud("../sample_data/plys/0.ply")  # target 为目标点云
source_pts.paint_uniform_color([1, 0.706, 0])  # source 为黄色
target_pts.paint_uniform_color([0, 0.651, 0.929])  # target 为蓝色
print(source_pts, target_pts)
processed_source, outlier_index = source_pts.remove_radius_outlier(nb_points=16,
                                                                   radius=0.5)

processed_target, outlier_index = target_pts.remove_radius_outlier(nb_points=16,
                                                                   radius=0.5)
print(processed_target)
o3d.visualization.draw_geometries([processed_source, processed_target])
