# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/10/18 21:02
# @FileName : icp配准.py
# @Software : PyCharm
# ICP (iterative closest point), 是对点云配准目前最常用的方法。
# 原理就是不断的对一个点云进行转换，并计算其中每个点与另外一个点云集的距离，将它转换成一个 fitness score。
# 然后不断地变换知道将这个总距离降到最低。一般来说icp都是经过全局配准之后运用的，也就是说两个点云要先被粗略地配准，然后icp会完成配准的微调。
import numpy as np
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
threshold = 1.0  # 移动范围的阀值
# 设置初始变换矩阵 类似于单应变换矩阵
trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                         [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                         [0, 0, 1, 0],  # 这个矩阵为初始变换
                         [0, 0, 0, 1]])
reg_p2p = o3d.pipelines.registration.registration_icp(processed_source, processed_target, threshold, trans_init,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
processed_source.transform(reg_p2p.transformation)
vis = o3d.visualization.Visualizer()
vis.create_window()

# 将两个点云放入visualizer
vis.add_geometry(processed_source)
vis.add_geometry(processed_target)

# 让visualizer渲染点云
vis.update_geometry(processed_source)
vis.update_geometry(processed_target)

vis.poll_events()
vis.update_renderer()

vis.run()
