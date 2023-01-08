# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/10/18 20:08
# @FileName : 点云ICP配准.py
# @Software : PyCharm
import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


source_pts = o3d.io.read_point_cloud(filename="../sample_data/plys/6.ply")  # source 为需要配准的点云
target = o3d.io.read_point_cloud("../sample_data/plys/0.ply")  # target 为目标点云
source_pts.paint_uniform_color([1, 0.706, 0])  # source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])  # target 为蓝色
print(source_pts)
print(target)
vis = o3d.visualization.Visualizer()
vis.create_window()

# 将两个点云放入visualizer
vis.add_geometry(source_pts)
vis.add_geometry(target)

# 让visualizer渲染点云
vis.update_geometry(source_pts)
vis.update_geometry(target)

