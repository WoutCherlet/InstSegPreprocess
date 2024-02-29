import os
import glob
import sys

import numpy as np
import open3d as o3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import merge_pointclouds, read_pc

def test_train_split(understory_file, trees_folder, odir):

    understory_pc = read_pc(understory_file)

    max_bound = understory_pc.get_max_bound().numpy()
    min_bound = understory_pc.get_min_bound().numpy()

    bound_x_min, bound_y_min, _ = min_bound
    bound_x_max, bound_y_max, _ = max_bound

    train_odir = os.path.join(odir, "trees", "train")
    if not os.path.exists(train_odir):
        os.makedirs(train_odir)
    val_odir = os.path.join(odir, "trees", "val")
    if not os.path.exists(val_odir):
        os.makedirs(val_odir)
    test_odir = os.path.join(odir, "trees", "test")
    if not os.path.exists(test_odir):
        os.makedirs(test_odir)

    # Litchfield: x: 0 -> 100, y: -100 -> 0
    # divide training: right 60 % of plot, val+ test: left 40%
    # val: bottom 50% of left area, test: top 50 %
    test_val_x_max = bound_x_min + 2/5*(bound_x_max - bound_x_min)
    test_y_min = bound_y_min + 1/2*(bound_y_max - bound_y_min)

    test_trees = []
    val_trees = []
    train_trees = []

    print("Dividing trees")
    i = 1
    for file in glob.glob(os.path.join(trees_folder, "*.ply")):
        filename = os.path.basename(file)
        tree = read_pc(file)

        # clear attributes to make merging possible
        tree_l = tree.to_legacy()
        tree = o3d.t.geometry.PointCloud.from_legacy(tree_l, o3d.core.float64)

        # add labels of trees: semantic is 1, instances positive int starting at 0
        tree.point.semantic = o3d.core.Tensor(np.ones(len(tree.point.positions), dtype=np.int32)[:,None])
        tree.point.instance = o3d.core.Tensor((i+1)*np.ones(len(tree.point.positions), dtype=np.int32)[:,None])
        i += 1

        pcl_bbox = tree.get_axis_aligned_bounding_box()
        bbox_min = pcl_bbox.min_bound
        bbox_max = pcl_bbox.max_bound

        # any tree that overlaps with test/val/train areas is 'part' of that area, cut to bbox later anyway
        if bbox_min[0] < test_val_x_max:
            if bbox_min[1] < test_y_min:
                out_path = os.path.join(val_odir, filename)
                o3d.t.io.write_point_cloud(out_path, tree)
                val_trees.append(tree)
            if bbox_max[1] > test_y_min:
                out_path = os.path.join(test_odir, filename)
                o3d.t.io.write_point_cloud(out_path, tree)
                test_trees.append(tree)
        if bbox_max[0] > test_val_x_max:
            out_path = os.path.join(train_odir, filename)
            o3d.t.io.write_point_cloud(out_path, tree)
            train_trees.append(tree)
    

    # add terrain labels: semantic is 0, instance = -1
    print("Cutting areas")

    understory_pc.point.semantic = o3d.core.Tensor(np.zeros(len(understory_pc.point.positions), dtype=np.int32)[:,None])
    understory_pc.point.instance = o3d.core.Tensor((-1)*np.ones(len(understory_pc.point.positions), dtype=np.int32)[:,None])

    max_bound[2] = np.inf
    min_bound[2] = -np.inf

    # slice understory pointcloud into actual sizes
    test_max_bound = max_bound.copy()
    test_max_bound[0] = test_val_x_max
    test_min_bound = min_bound.copy()
    test_min_bound[1] = test_y_min

    val_max_bound = max_bound.copy()
    val_max_bound[0] = test_val_x_max
    val_max_bound[1] = test_y_min
    test_max_bound[2] = np.inf

    train_min_bound = min_bound.copy()
    train_min_bound[0] = test_val_x_max

    test_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = test_min_bound, max_bound = test_max_bound)
    val_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = min_bound, max_bound = val_max_bound)
    train_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = train_min_bound, max_bound = max_bound)

    test_sliced = understory_pc.crop(test_bbox)
    val_sliced = understory_pc.crop(val_bbox)
    train_sliced = understory_pc.crop(train_bbox)


    print("Merging trees and understory")

    # merge trees with
    test_trees_pc = merge_pointclouds(test_trees)
    val_trees_pc = merge_pointclouds(val_trees)
    train_trees_pc = merge_pointclouds(train_trees)

    test_plot_pc = test_trees_pc + test_sliced
    val_plot_pc = val_trees_pc + val_sliced
    train_plot_pc = train_trees_pc + train_sliced

    # re crop so trees are also cropped
    test_plot_pc = test_plot_pc.crop(test_bbox)
    val_plot_pc = val_plot_pc.crop(val_bbox)
    train_plot_pc = train_plot_pc.crop(train_bbox)

    print("Writing areas")

    # write out test, val and train plots as temp
    o3d.t.io.write_point_cloud(os.path.join(odir, "test_merged.ply"), test_plot_pc)
    o3d.t.io.write_point_cloud(os.path.join(odir, "val_merged.ply"), val_plot_pc)
    o3d.t.io.write_point_cloud(os.path.join(odir, "train_merged.ply"), train_plot_pc)

    return


def main():
    PLOT_PC = "/media/wcherlet/Stor1/wout/data/Litchfield/Augustus/understory/merged_pc.ply"
    TREES_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/Augustus/trees"
    ODIR = "/media/wcherlet/Stor1/wout/data/Litchfield/Augustus/litchfield_test_train"

    test_train_split(PLOT_PC, TREES_FOLDER, ODIR)


if __name__ == "__main__":
    main()