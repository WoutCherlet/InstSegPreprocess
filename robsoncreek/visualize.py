import os
import sys

import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_ply_folder


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

def main():
    plot_file = "/media/wcherlet/Stor1/wout/data/RobsonCreek/plot_pc/RC_2018_1cm_1ha_buffer.ply"
    trees_folder = "/media/wcherlet/Stor1/wout/data/RobsonCreek/tree_pcs"

    scanner_pos_ply = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/scanpositions2018.ply"

    plot = o3d.t.io.read_point_cloud(plot_file)
    trees = read_ply_folder(trees_folder)

    # read scanner positions
    scanner_pc = o3d.t.io.read_point_cloud(scanner_pos_ply)
    scanner_pos = scanner_pc.point.positions.numpy()

    plot_layout(plot, trees, scanner_pos)

    # trees["plot/plot_pc.ply"] = plot
    # ds_visualization(trees, odir="/media/wcherlet/Stor1/wout/data/RobsonCreek/vis_ds")
    # ds_visualization(trees, odir="/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/vis_ds")


    return

if __name__ == "__main__":
    main()