import glob
import os

import numpy as np
import open3d as o3d

from tree_io import read_ply_folder


def split_quadrants(plot_file):

    # split the plot into quadrants for segmentation

    plot_pc = o3d.io.read_point_cloud(plot_file)

    bbox_plot = plot_pc.get_axis_aligned_bounding_box()

    plot_min_bound = bbox_plot.get_min_bound()
    plot_max_bound = bbox_plot.get_max_bound()

    max_bound_a = np.array([0,0,plot_max_bound[2]])
    bbox_a = o3d.geometry.AxisAlignedBoundingBox(min_bound=plot_min_bound, max_bound = max_bound_a)

    max_bound_b = np.array([0,plot_max_bound[1], plot_max_bound[2]])
    min_bound_b = np.array([plot_min_bound[0], 0, plot_min_bound[2]])
    bbox_b = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_b, max_bound = max_bound_b)

    max_bound_c = np.array([plot_max_bound[0], 0, plot_max_bound[2]])
    min_bound_c = np.array([0, plot_min_bound[1], plot_min_bound[2]])
    bbox_c = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_c, max_bound = max_bound_c)

    max_bound_d = np.array([plot_max_bound[0], plot_max_bound[1], plot_max_bound[2]])
    min_bound_d = np.array([0, 0, plot_min_bound[2]])
    bbox_d = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_d, max_bound = max_bound_d)

    quad_a = plot_pc.crop(bbox_a)
    quad_b = plot_pc.crop(bbox_b)
    quad_c = plot_pc.crop(bbox_c)
    quad_d = plot_pc.crop(bbox_d)

    odir = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/segmented_distance/1cm/quadrants/"
    
    if not os.path.exists(odir):
        os.makedirs(odir)

    o3d.io.write_point_cloud(os.path.join(odir, "quad_a.ply"), quad_a)
    o3d.io.write_point_cloud(os.path.join(odir, "quad_b.ply"), quad_b)
    o3d.io.write_point_cloud(os.path.join(odir, "quad_c.ply"), quad_c)
    o3d.io.write_point_cloud(os.path.join(odir, "quad_d.ply"), quad_d)

    return



def trees_in_quadrant(quadrant_file, tree_folder, odir):

    if not os.path.exists(odir):
        os.makedirs(odir)

    quadrant_pc = o3d.t.io.read_point_cloud(quadrant_file)
    tree_dict = read_ply_folder(tree_folder)

    quadrant_bbox = quadrant_pc.get_axis_aligned_bounding_box()

    for k in tree_dict:
        tree_pc = tree_dict[k]

        in_idx = quadrant_bbox.get_point_indices_within_bounding_box(tree_pc.point.positions)

        n_in = in_idx.numpy().shape[0]
        if n_in != 0:
            o3d.t.io.write_point_cloud(os.path.join(odir, k), tree_pc)
    
    return

def trees_quadrants(quadrants_dir, trees_dir):
    odir = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/segmented_distance/1cm/quadrants/"

    for quadrant in glob.glob(os.path.join(quadrants_dir, "*.ply")):
        odir_quadrant = os.path.join(odir, quadrant[:-4])

        trees_in_quadrant(quadrant, trees_dir, odir_quadrant)


def main():
    plot_file = "/media/wcherlet/Stor1/wout/data/RobsonCreek/plot_pc/RC_2018_1cm_1ha_buffer.ply"
    trees_folder = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/segmented_distance/1cm/segmented_trees"

    plot = o3d.t.io.read_point_cloud(plot_file)
    trees = read_ply_folder(trees_folder)

    # plot_file = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/segmented_distance/1cm/leftover_points.ply"

    # split_quadrants(plot_file)

    quadrants_dir = "/media/wcherlet/SSD WOUT/BenchmarkPaper/RobsonCreek/segmented_distance/1cm/quadrants"

    trees_quadrants(quadrants_dir, trees_folder)

    return

if __name__ == "__main__":
    main()

