import os
import time

import numpy as np
import open3d as o3d


def read_pointclouds(folder):
    out = {}

    for file in os.listdir(folder):
        if not file[-4:] == ".ply":
            print(file)
            print("not ply")
            continue
        # read like this to delete custom attributes
        pc = o3d.io.read_point_cloud(os.path.join(folder, file))
        # remove colors if present, otherwise merge no work
        pc.colors = o3d.utility.Vector3dVector()
        pc = o3d.t.geometry.PointCloud.from_legacy(pc)
        out[file] = pc

    return out

def overlap_trees_distance(plot_pc, tree_folder, dist_threshold=0.1, odir = None, save_tree_sections = False, backup_iterations=None):
    '''
    Label points in plot_pc according to tree instances in tree_folder
    Labeled pointcloud is written to odir/labeled_plot.ply
    
    Args:
        plot_path (string): 
            path to pointcloud of entire plot (type must be readable by open3d)
        tree_folder (string): 
            path to folder containing individual tree pointclouds (type must be readable by open3d)
        dist_threshold (float): 
            distance threshold for tree point classification 
        save_tree_sections (bool): 
            if true, labeled sections of the plot inside the bounding box of each tree are outputted
        backup_iterations (int): 
            if set, backs up the labeled plot count every backup_iterations iterations

    Returns:
        None
    '''

    if odir is None:
        odir = os.getcwd()

    if not os.path.exists(odir):
        os.makedirs(odir)

    if save_tree_sections:
        odir_single_trees = os.path.join(odir, "bbox_trees")
        if not os.path.exists(odir_single_trees):
            os.makedirs(odir_single_trees)

    plot_pc = o3d.io.read_point_cloud(plot_pc)
    tree_dict = read_pointclouds(tree_folder)
    instance_labels_plot = np.zeros(len(plot_pc.points), dtype=np.int32)
    i=1

    for k in tree_dict:
        print(f"({time.strftime('%Y-%m-%d %H:%M:%S')}) processing {k} ({i}/{len(tree_dict)})")
        tree_pc = tree_dict[k]
        tree_pc_legacy = tree_pc.to_legacy()

        tree_bbox = tree_pc_legacy.get_axis_aligned_bounding_box()
        # small buffer around tree
        tree_bbox.max_bound = tree_bbox.max_bound + np.array([dist_threshold+0.01, dist_threshold+0.01, dist_threshold+0.01])
        tree_bbox.min_bound = tree_bbox.min_bound - np.array([dist_threshold+0.01, dist_threshold+0.01, dist_threshold+0.01])

        # get points in bbox around tree
        inliers_indices = tree_bbox.get_point_indices_within_bounding_box(plot_pc.points)
        inliers_pc = plot_pc.select_by_index(inliers_indices, invert=False)

        # for points in bbox, calculate distance to tree and get their indices
        distances = inliers_pc.compute_point_cloud_distance(tree_pc_legacy)
        distances = np.asarray(distances)
        tree_ind = np.where(distances < dist_threshold)[0]

        # label original instance array
        inliers_indices = np.asarray(inliers_indices)
        plot_indices_tree = inliers_indices[tree_ind] # map mask on inlier indices to mask on original indices
        instance_labels_plot[plot_indices_tree] = i
        i += 1

        # optionally save tree sections for easier visualization
        if save_tree_sections:
            inliers_instance_labels = np.zeros(len(inliers_pc.points), dtype=np.int32)
            inliers_instance_labels[tree_ind] = i
            inliers_pc_t = o3d.t.geometry.PointCloud.from_legacy(inliers_pc)
            inliers_pc_t.point.instance = o3d.core.Tensor(inliers_instance_labels[:,np.newaxis])
            o3d.t.io.write_point_cloud(os.path.join(odir_single_trees, f"labeled_bbox_{k[-14:-4]}.ply"), inliers_pc_t)

        # backup after backup_interval
        if isinstance(backup_iterations, int) and i % backup_iterations == 0:
            # back up results untill now
            plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
            o3d.t.io.write_point_cloud(os.path.join(odir, f"plot_labeled_{i}_instances.ply"), plot_pc)
        
    
    plot_pc.point.instance = o3d.core.Tensor(instance_labels_plot[:, np.newaxis])
    o3d.t.io.write_point_cloud(os.path.join(odir, "plot_labeled.ply"), plot_pc)

    return


def main():
    # TODO: edit here or change to argparse
    PLOT_FILE = ""
    TREES_FOLDER = ""
    ODIR = ""

    overlap_trees_distance(PLOT_FILE, TREES_FOLDER, dist_threshold=0.1, odir=ODIR, save_tree_sections=True, backup_iterations=None)



if __name__ == '__main__':
    main()