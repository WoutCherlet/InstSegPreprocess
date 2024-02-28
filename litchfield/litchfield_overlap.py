import os
import sys
import glob
import time

import open3d as o3d
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_pc, merge_pointclouds



def overlap_distance(plot_pc, tree_dict, odir, distance_th=0.1):
    # SAME FUNCTION AS RC, but dont write out trees

    # for each tree:
    # cut plot to bounding box + small buffer of couple cm
    # calculate distance of each point in cut plot to tree
    # if dist smaller than tolerance (1 cm or so): label point as tree
    # add labels to entire plot pc using cut indices (hard part but should be possible)

    if not os.path.exists(odir):
        os.makedirs(odir)

    odir_bbox_trees = os.path.join(odir, "bbox_trees")
    if not os.path.exists(odir_bbox_trees):
        os.makedirs(odir_bbox_trees)

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

        # label original instance array
        inliers_indices = np.asarray(inliers_indices)
        plot_indices_tree = inliers_indices[tree_ind] # map mask on inlier indices to mask on original indices
        instance_labels_plot[plot_indices_tree] = i
        leftover_mask[plot_indices_tree] = 0
        i += 1

        # DEBUG: write smaller pcs for debug
        inliers_instance_labels = np.zeros(len(inliers_pc.points), dtype=np.int32)
        inliers_instance_labels[tree_ind] = i
        inliers_pc_t = o3d.t.geometry.PointCloud.from_legacy(inliers_pc)
        inliers_pc_t.point.instance = o3d.core.Tensor(inliers_instance_labels[:,np.newaxis])
        o3d.t.io.write_point_cloud(os.path.join(odir_bbox_trees, f"labeled_bbox_{k[-14:-4]}.ply"), inliers_pc_t)


    leftover_pc = plot_pc_legacy.select_by_index(np.nonzero(leftover_mask)[0])
    o3d.io.write_point_cloud(os.path.join(odir, "leftover_points.ply"), leftover_pc)
    # leftover_pc_t = o3d.t.geometry.PointCloud.from_legacy(leftover_pc)
    # o3d.t.io.write_point_cloud(os.path.join(odir, "leftover_points.ply"), leftover_pc_t)

    # TODO: segfault when writing pointcloud using tensor version, writing using legacy but takes double space and does not save labels!
    plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
    # plot_legacy = plot_pc.to_legacy()
    # o3d.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_legacy)
    o3d.t.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_pc)

    return


def get_tile_names(trees_dir):
    segmented_tiles = []
    for dir in glob.glob(os.path.join(trees_dir, "*")):
        segmented_tiles.append(os.path.basename(dir)[:-6])
    return segmented_tiles

def read_tiles(tile_dir, tile_names):
    tiles = []
    for tile_name in tile_names:
        tile = read_pc(os.path.join(tile_dir, tile_name+".las"))
        tiles.append(tile)
    return tiles

def read_trees(FOLDER, prefix):
    out = {}
    for file in glob.glob(os.path.join(FOLDER, prefix+ "*.ply")):
        if not file[-4:] == ".ply":
            print(file)
            print("not ply")
            continue
        # read like this to delete custom attributes
        pc = o3d.io.read_point_cloud(file)
        # remove colors if present, otherwise merge no work
        pc.colors = o3d.utility.Vector3dVector()
        pc = o3d.t.geometry.PointCloud.from_legacy(pc)
        out[os.path.basename(file)] = pc
    return out


def tiles_to_pc(tile_folder):

    pcs = []
    for tile in glob.glob(os.path.join(tile_folder, "*.txt")):
        pc = read_pc(tile)
        pcs.append(pc)

    merged_pc = merge_pointclouds(pcs)
    o3d.t.io.write_point_cloud(os.path.join(tile_folder, "merged_pc.ply"), merged_pc)
    return



def main():


    # ----------------------------
    # OVERLAP TILES
    
    # NOTE: this was only necessary for September trees as they havent been manually segmented
    # but august trees are manual and clean enough so skip this

    # TILE_DIR = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/TILES_litchfield_sept_all36comb_AOI_dev50_refl-15_1cm/"
    # TREES_DIR = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Bomen/"
    # ALL_TREES_DIR = "/media/wcherlet/Stor1/wout/data/Litchfield/September/trees"
    # ODIR = "/media/wcherlet/Stor1/wout/data/Litchfield/September/reclassified_5cm"

    # tile_names = get_tile_names(TREES_DIR)
    # print(tile_names)
    # tiles = read_tiles(TILE_DIR, tile_names)
    # trees = read_trees(ALL_TREES_DIR, prefix=tile_names[0])

    # TODO: TEMP : run for single tile
    # overlap_distance(tiles[0], trees, ODIR, distance_th=0.05)

    # ----------------------------
    # MERGE UNDERSTORY TILES

    TILE_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/Augustus/understory"

    tiles_to_pc (TILE_FOLDER)




    return

if __name__ == "__main__":
    main()