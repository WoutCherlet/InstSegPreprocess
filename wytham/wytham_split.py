import os
import numpy as np
import open3d as o3d

from tree_io import merge_pointclouds

def test_train_split(tree_tiles_dict, understory_tiles_dict, trees_folder, odir):
    pointclouds = list(tree_tiles_dict.values())
    pointclouds += list(understory_tiles_dict.values())

    merged_pointcloud = merge_pointclouds(pointclouds)

    max_bound = merged_pointcloud.get_max_bound().numpy()
    min_bound = merged_pointcloud.get_min_bound().numpy()

    bound_x_min = min_bound[0]
    bound_x_max = max_bound[0]

    train_odir = os.path.join(odir, "trees", "train")
    if not os.path.exists(train_odir):
        os.makedirs(train_odir)
    val_odir = os.path.join(odir, "trees", "val")
    if not os.path.exists(val_odir):
        os.makedirs(val_odir)
    test_odir = os.path.join(odir, "trees", "test")
    if not os.path.exists(test_odir):
        os.makedirs(test_odir)

    
    # Make mapping of tree file to unique number 
    filenames = [f for f in os.listdir(trees_folder) if f[-3:] == 'ply']

    test_x_max = bound_x_min + 1/5*(bound_x_max - bound_x_min)
    val_x_max = test_x_max + 1/5*(bound_x_max - bound_x_min)

    test_trees = []
    val_trees = []
    train_trees = []

    print("Dividing trees")

    for i, filename in enumerate(filenames):
        pcl = o3d.t.io.read_point_cloud(os.path.join(trees_folder, filename))

        # add labels of trees: semantic is 1, instances positive int starting at 0

        pcl.point.semantic = o3d.core.Tensor(np.ones(len(pcl.point.positions), dtype=np.int32)[:,None])
        pcl.point.instance = o3d.core.Tensor((i+1)*np.ones(len(pcl.point.positions), dtype=np.int32)[:,None])

        pcl_bbox = pcl.get_axis_aligned_bounding_box()

        bbox_min = pcl_bbox.min_bound
        bbox_max = pcl_bbox.max_bound

        # any tree that overlaps with test/val/train areas is
        if bbox_min[0] < test_x_max:
            out_path = os.path.join(test_odir, filename)
            o3d.t.io.write_point_cloud(out_path, pcl)
            test_trees.append(pcl)
        if bbox_min[0] < val_x_max and bbox_max[0] > test_x_max:
            out_path = os.path.join(val_odir, filename)
            o3d.t.io.write_point_cloud(out_path, pcl)
            val_trees.append(pcl)
        if bbox_max[0] > val_x_max:
            out_path = os.path.join(train_odir, filename)
            o3d.t.io.write_point_cloud(out_path, pcl)
            train_trees.append(pcl)


    
    # merge all understory tiles that overlap with test, val and train areas
    test_tiles = []
    val_tiles = []
    train_tiles = []

    print("Dividing understory tiles")
    
    for tile in understory_tiles_dict:
        pc = understory_tiles_dict[tile]
        min_bound_x, _, _ = pc.get_min_bound().numpy()
        max_bound_x, _, _ = pc.get_max_bound().numpy()

        if min_bound_x < test_x_max:
            test_tiles.append(pc)
        if min_bound_x < val_x_max and max_bound_x > test_x_max:
            val_tiles.append(pc)
        if max_bound_x > val_x_max:
            train_tiles.append(pc)

    test_merged = merge_pointclouds(test_tiles)
    val_merged = merge_pointclouds(val_tiles)
    train_merged = merge_pointclouds(train_tiles)

    # add terrain labels: semantic is 0, instance = -1

    test_merged.point.semantic = o3d.core.Tensor(np.zeros(len(test_merged.point.positions), dtype=np.int32)[:,None])
    test_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(test_merged.point.positions), dtype=np.int32)[:,None])
    val_merged.point.semantic = o3d.core.Tensor(np.zeros(len(val_merged.point.positions), dtype=np.int32)[:,None])
    val_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(val_merged.point.positions), dtype=np.int32)[:,None])
    train_merged.point.semantic = o3d.core.Tensor(np.zeros(len(train_merged.point.positions), dtype=np.int32)[:,None])
    train_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(train_merged.point.positions), dtype=np.int32)[:,None])

    print("Merging trees and understory")

    # merge trees
    test_trees_pc = merge_pointclouds(test_trees)
    val_trees_pc = merge_pointclouds(val_trees)
    train_trees_pc = merge_pointclouds(train_trees)

    test_plot_pc = test_trees_pc + test_merged
    val_plot_pc = val_trees_pc + val_merged
    train_plot_pc = train_trees_pc + train_merged

    print("Cutting areas")

    # slice merged test, val and train pointclouds into actual sizes
    test_max_bound = max_bound.copy()
    test_max_bound[0]  = test_x_max

    val_max_bound = max_bound.copy()
    val_max_bound[0] = val_x_max
    val_min_bound = min_bound.copy()
    val_min_bound[0] = test_x_max

    train_min_bound = min_bound.copy()
    train_min_bound[0] = val_x_max

    test_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = min_bound, max_bound = test_max_bound)
    val_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = val_min_bound, max_bound = val_max_bound)
    train_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = train_min_bound, max_bound = max_bound)

    test_sliced = test_plot_pc.crop(test_bbox)
    val_sliced = val_plot_pc.crop(val_bbox)
    train_sliced = train_plot_pc.crop(train_bbox)

    print("Writing areas")

    # write out test, val and train plots as temp
    o3d.t.io.write_point_cloud(os.path.join(odir, "test_merged.ply"), test_sliced)
    o3d.t.io.write_point_cloud(os.path.join(odir, "val_merged.ply"), val_sliced)
    o3d.t.io.write_point_cloud(os.path.join(odir, "train_merged.ply"), train_sliced)

    return