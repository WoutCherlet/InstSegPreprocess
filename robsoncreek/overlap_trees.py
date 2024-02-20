import os
import sys
import csv
import time

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# TODO: TEMP: actual scripts should be in seperate folder 
# guido (python papa) says its an antipattern to have script inside package
# but I just want some scripts that use some common utils
# so fuck guido i guess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_pointclouds, merge_pointclouds

from wytham.seperate_trees import isclose_nd


def ds_visualization(pointclouds, odir):
    if not os.path.exists(odir):
        os.makedirs(odir)
    if not os.path.exists(os.path.join(odir, "plot")):
        os.makedirs(os.path.join(odir, "plot"))
    for file_name in pointclouds:
        pc = pointclouds[file_name]
        ds = pc.voxel_down_sample(0.1)
        o3d.t.io.write_point_cloud(os.path.join(odir, file_name), ds)
    return


def bbox_to_patch(bbox):
    points = bbox.get_box_points().numpy()
    x_min, y_min, _ = np.min(points, axis=0)
    extent_x, extent_y = bbox.get_extent().numpy()[:2]
    rect = patches.Rectangle((x_min, y_min), extent_x, extent_y, linewidth=1, edgecolor='r', facecolor='none')
    return rect

def plot_layout(plot_pc, tree_dict, scanner_pos):

    # plot size
    max_plot = plot_pc.get_max_bound().numpy()
    min_plot = plot_pc.get_min_bound().numpy()

    print("Plot size:")
    print(f"Max bound: {max_plot}")
    print(f"Min bound: {min_plot}")
    print(f"Size: {max_plot - min_plot}")

    # trees bbox

    # all_trees_pc = merge_pointclouds(list(tree_dict.values()))

    # max_trees = all_trees_pc.get_max_bound().numpy()
    # min_trees = all_trees_pc.get_min_bound().numpy()

    # print("Trees bbox size:")
    # print(f"Max bound: {max_trees}")
    # print(f"Min bound: {min_trees}")
    # print(f"Size: {max_trees - min_trees}")

    # get top down 2d view of plot and segmented trees

    bbox_plot = plot_pc.get_axis_aligned_bounding_box()
    rect = bbox_to_patch(bbox_plot)

    fig, ax = plt.subplots()
    ax.add_patch(rect)

    scatter_x = []
    scatter_y = []

    for k in tree_dict:
        tree = tree_dict[k]
        bbox = tree.get_axis_aligned_bounding_box()

        rect = bbox_to_patch(bbox)

        tree_center = tree.get_center().numpy()

        scatter_x.append(tree_center[0])
        scatter_y.append(tree_center[1])
        ax.add_patch(rect)

    ax.scatter(scatter_x, scatter_y, color='red')

    rect_50 = patches.Rectangle((-25,-25), 50, 50, linewidth=3, edgecolor='green', facecolor='none')
    rect_70 = patches.Rectangle((-35,-35), 70, 70, linewidth=3, edgecolor='green', facecolor='none')

    ax.add_patch(rect_50)
    ax.add_patch(rect_70)

    ax.scatter(scanner_pos[:,0], scanner_pos[:,1], color='black')

    plt.show()

    return


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


def overlap_trees_distance(plot_pc, tree_dict):

    # for each tree:
    # cut plot to bounding box + small buffer of couple cm
    # calculate distance of each point in cut plot to tree
    # if dist smaller than tolerance (1 cm or so): label point as tree
    # add labels to entire plot pc using cut indices (hard part but should be possible)

    odir = "/media/wcherlet/Stor1/wout/data/RobsonCreek/segmented_distance/"
    if not os.path.exists(odir):
        os.makedirs(odir)

    odir_single_trees = os.path.join(odir, "bbox_trees")
    if not os.path.exists(odir_single_trees):
        os.makedirs(odir_single_trees)

    plot_pc_legacy = plot_pc.to_legacy()
    i=1
    instance_labels_plot = np.zeros(len(plot_pc_legacy.points), dtype=np.int32)

    for k in tree_dict:
        print(f"({time.strftime('%Y-%m-%d %H:%M:%S')}) processing {k} ({i}/{len(tree_dict)})")
        tree_pc = tree_dict[k]
        tree_pc_legacy = tree_pc.to_legacy()

        tree_bbox = tree_pc_legacy.get_axis_aligned_bounding_box()
        # small buffer around tree
        tree_bbox.max_bound = tree_bbox.max_bound + np.array([0.11, 0.11, 0.11])
        tree_bbox.min_bound = tree_bbox.min_bound - np.array([0.11, 0.11, 0.11])

        # get points in bbox around tree
        inliers_indices = tree_bbox.get_point_indices_within_bounding_box(plot_pc_legacy.points)
        inliers_pc = plot_pc_legacy.select_by_index(inliers_indices, invert=False)

        # for points in bbox, calculate distance to tree and get their indices
        distances = inliers_pc.compute_point_cloud_distance(tree_pc_legacy)
        distances = np.asarray(distances)
        tree_ind = np.where(distances < 0.1)[0]

        # label original instance array
        inliers_indices = np.asarray(inliers_indices)
        plot_indices_tree = inliers_indices[tree_ind] # map mask on inlier indices to mask on original indices
        instance_labels_plot[plot_indices_tree] = i
        i += 1

        # TODO: TEMP: write smaller pcs for debug
        inliers_instance_labels = np.zeros(len(inliers_pc.points), dtype=np.int32)
        inliers_instance_labels[tree_ind] = i
        inliers_pc_t = o3d.t.geometry.PointCloud.from_legacy(inliers_pc)
        inliers_pc_t.point.instance = o3d.core.Tensor(inliers_instance_labels[:,np.newaxis])
        o3d.t.io.write_point_cloud(os.path.join(odir_single_trees, f"labeled_bbox_{k[-14:-4]}.ply"), inliers_pc_t)

        # TODO: TEMP: backup every 20 instances
        if i % 20 == 0:
            # back up results untill now
            plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
            o3d.t.io.write_point_cloud(os.path.join(odir, f"plot_labeled_{i}_instances.ply"), plot_pc)
        

    plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
    o3d.t.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_pc)

    return


def main():

    plot_file = "/media/wcherlet/Stor1/wout/data/RobsonCreek/plot_pc/RC_2018_2cm_1ha_10mbuffer.ply"
    trees_folder = "/media/wcherlet/Stor1/wout/data/RobsonCreek/tree_pcs"

    # plot_file = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/plot_pc/RC_2018_2cm_1ha_10mbuffer.ply"
    # trees_folder = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/tree_pcs"

    # plot_file = "/media/wcherlet/Stor1/wout/data/RobsonCreek/vis_ds/plot/plot_pc.ply"
    # trees_folder = "/media/wcherlet/Stor1/wout/data/RobsonCreek/vis_ds"

    # plot_file = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/vis_ds/plot/plot_pc.ply"
    # trees_folder = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/vis_ds"

    # scanner_pos_csv = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/RC_2018.csv"
    # scanner_pos_ply = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/scanpositions2018.ply"

    plot = o3d.t.io.read_point_cloud(plot_file)
    trees = read_pointclouds(trees_folder)

    # read scanner positions
    # with open(scanner_pos_csv, 'r') as f:
    #     csv_list = list(csv.reader(f, delimiter=","))

    # scanner_pos = np.array(csv_list)[1:,1:4]
    # scanner_pos = scanner_pos.astype(float)

    # read scanner positions
    # scanner_pc = o3d.t.io.read_point_cloud(scanner_pos_ply)
    # scanner_pos = scanner_pc.point.positions.numpy()

    # plot_layout(plot, trees, scanner_pos)

    # trees["plot/plot_pc.ply"] = plot
    # ds_visualization(trees, odir="/media/wcherlet/Stor1/wout/data/RobsonCreek/vis_ds")
    # ds_visualization(trees, odir="/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/vis_ds")

    overlap_trees_distance(plot, trees)

    return



if __name__ == "__main__":

    main()