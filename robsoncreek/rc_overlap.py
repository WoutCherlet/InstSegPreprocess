import os
import sys
import csv
import time

import numpy as np
import open3d as o3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_ply_folder

from wytham.seperate_trees import isclose_nd


def overlap_trees_naive(plot_pc, tree_dict):

    # NOTE: very bad
    assert False, "Don't use this it sucks"

    # loop through trees, check overlapping points in plot pc and mark as instance

    odir = "/media/wcherlet/Stor1/wout/data/RobsonCreek/segmented/"
    if not os.path.exists(odir):
        os.makedirs(odir)

    plot_points = plot_pc.point.positions.numpy()

    total_tree_mask = np.zeros(len(plot_points), dtype=np.int32)
    # total_tree_mask_exact = np.zeros(len(plot_points), dtype=np.int32)
    i=1
    instance_label_plot = np.zeros(len(plot_points), dtype=np.int32)
    for k in tree_dict:
        print(f"processing {k} ({i}/{len(tree_dict)})")

        tree_pc = tree_dict[k]
        tree_points = tree_pc.point.positions.numpy()

        row_match = isclose_nd(plot_points, np.asarray(tree_points))
        # row_match_exact = isin_nd(plot_points, np.asarray(tree_points))

        # get total tree mask for total
        total_tree_mask = np.logical_or(total_tree_mask, row_match)
        # total_tree_mask_exact = np.logical_or(total_tree_mask_exact, row_match_exact)

        # update label array with instance ID
        instance_label_plot[row_match] = i

        if i % 10 == 0:
            # back up results untill now
            plot_pc.point.instance = o3d.core.Tensor(instance_label_plot[:, np.newaxis])
            o3d.t.io.write_point_cloud(os.path.join(odir, f"plot_labeled_{i}_instances.ply"), plot_pc)

        i += 1


    plot_pc.point.instance = o3d.core.Tensor(instance_label_plot[:, np.newaxis])
    o3d.t.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_pc)
        
    understory_mask = np.logical_not(total_tree_mask)
    understory_points = plot_points[understory_mask]
    understory_cloud = o3d.t.geometry.PointCloud()
    understory_cloud.point.positions = o3d.core.Tensor(understory_points)
    o3d.t.io.write_point_cloud(os.path.join(odir, "understory.ply"), understory_cloud)
    

    return


def overlap_trees_distance(plot_pc, tree_dict, distance_th=0.1):

    # for each tree:
    # cut plot to bounding box + small buffer of couple cm
    # calculate distance of each point in cut plot to tree
    # if dist smaller than tolerance (1 cm or so): label point as tree
    # add labels to entire plot pc using cut indices (hard part but should be possible)

    odir = "/media/wcherlet/Stor1/wout/data/RobsonCreek/segmented_distance/1cm"
    if not os.path.exists(odir):
        os.makedirs(odir)

    odir_bbox_trees = os.path.join(odir, "bbox_trees")
    if not os.path.exists(odir_bbox_trees):
        os.makedirs(odir_bbox_trees)

    odir_segmented_trees = os.path.join(odir, "segmented_trees")
    if not os.path.exists(odir_segmented_trees):
        os.makedirs(odir_segmented_trees)

    plot_pc_legacy = plot_pc.to_legacy()
    i=1
    instance_labels_plot = np.zeros(len(plot_pc_legacy.points), dtype=np.int32)
    leftover_mask = np.ones(len(plot_pc_legacy.points), dtype=np.int32)

    for k in tree_dict:
        print(f"({time.strftime('%Y-%m-%d %H:%M:%S')}) processing {k} ({i}/{len(tree_dict)})")
        tree_pc = tree_dict[k]
        tree_pc_legacy = tree_pc.to_legacy()

        tree_bbox = tree_pc_legacy.get_axis_aligned_bounding_box()
        # small buffer around tree
        tree_bbox.max_bound = tree_bbox.max_bound + np.array([distance_th+0.01, distance_th+0.01, distance_th+0.01])
        tree_bbox.min_bound = tree_bbox.min_bound - np.array([distance_th+0.01, distance_th+0.01, distance_th+0.01])

        # get points in bbox around tree
        inliers_indices = tree_bbox.get_point_indices_within_bounding_box(plot_pc_legacy.points)
        inliers_pc = plot_pc_legacy.select_by_index(inliers_indices, invert=False)

        # for points in bbox, calculate distance to tree and get their indices
        distances = inliers_pc.compute_point_cloud_distance(tree_pc_legacy)
        distances = np.asarray(distances)
        tree_ind = np.where(distances < distance_th)[0]

        # Save segmented tree to folder
        tree_pc_segmented = inliers_pc.select_by_index(tree_ind, invert=False)
        tree_pc_seg_t = o3d.t.geometry.PointCloud.from_legacy(tree_pc_segmented)
        o3d.t.io.write_point_cloud(os.path.join(odir_segmented_trees, f"{k[-14:-4]}.ply"), tree_pc_seg_t)

        # label original instance array
        inliers_indices = np.asarray(inliers_indices)
        plot_indices_tree = inliers_indices[tree_ind] # map mask on inlier indices to mask on original indices
        instance_labels_plot[plot_indices_tree] = i
        leftover_mask[plot_indices_tree] = 0
        i += 1

        # TODO: TEMP: write smaller pcs for debug
        inliers_instance_labels = np.zeros(len(inliers_pc.points), dtype=np.int32)
        inliers_instance_labels[tree_ind] = i
        inliers_pc_t = o3d.t.geometry.PointCloud.from_legacy(inliers_pc)
        inliers_pc_t.point.instance = o3d.core.Tensor(inliers_instance_labels[:,np.newaxis])
        o3d.t.io.write_point_cloud(os.path.join(odir_bbox_trees, f"labeled_bbox_{k[-14:-4]}.ply"), inliers_pc_t)

        # TODO: TEMP: backup every 20 instances
        # if i % 20 == 0:
        #     # back up results untill now
        #     plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
        #     o3d.t.io.write_point_cloud(os.path.join(odir, f"plot_labeled_{i}_instances.ply"), plot_pc)


    # TODO: segfault when writing pointcloud using tensor version, writing using legacy but takes double space
    
    leftover_pc = plot_pc_legacy.select_by_index(np.nonzero(leftover_mask)[0])
    o3d.io.write_point_cloud(os.path.join(odir, "leftover_points.ply"), leftover_pc)
    # leftover_pc_t = o3d.t.geometry.PointCloud.from_legacy(leftover_pc)
    # o3d.t.io.write_point_cloud(os.path.join(odir, "leftover_points.ply"), leftover_pc_t)

    plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
    plot_legacy = plot_pc.to_legacy()
    o3d.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_legacy)
    # o3d.t.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_pc)

    return


def main():
    plot_file = "/media/wcherlet/Stor1/wout/data/RobsonCreek/plot_pc/RC_2018_1cm_1ha_buffer.ply"
    trees_folder = "/media/wcherlet/Stor1/wout/data/RobsonCreek/tree_pcs"

    plot = o3d.t.io.read_point_cloud(plot_file)
    trees = read_ply_folder(trees_folder)

    # overlap_trees_distance(plot, trees, distance_th=0.05)


    return

if __name__ == "__main__":
    main()