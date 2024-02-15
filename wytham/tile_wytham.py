import os
import open3d as o3d

from tree_io import read_pointclouds
from test_train_split import trees_in_plot, tile_area, test_train_split


def tile_wytham(merged_area_dir):

    # training area

    print("Tiling training area with overlap 5")

    odir = os.path.join(merged_area_dir, "tiles", "train")
    if not os.path.exists(odir):
        os.makedirs(odir)
    train_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "train_merged.ply"))

    tile_area(train_pc, x_n=6, y_n=11, plot_name="Wytham", odir=odir)

    # val area

    print("Tiling validation area with overlap 5")

    odir = os.path.join(merged_area_dir, "tiles", "val")
    if not os.path.exists(odir):
        os.makedirs(odir)
    validation_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "val_merged.ply"))

    tile_area(validation_pc, x_n=2, y_n=11, plot_name="Wytham", odir=odir)

    # test area


    odir = os.path.join(merged_area_dir, "tiles", "test")
    if not os.path.exists(odir):
        os.makedirs(odir)
    trees_odir = os.path.join(merged_area_dir, "trees", "test")

    test_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "test_merged.ply"))

    # for test: also divide trees into eval and non-eval trees
    test_trees = read_pointclouds(trees_odir)
    odir_trees = os.path.join(merged_area_dir, "test_trees_thresholded")
    trees_in_plot(test_pc, test_trees, odir_trees, threshold=0.9, output_all=True)


    print("Tiling testing area with overlap 5")

    tile_area(test_pc, x_n=2, y_n=11, plot_name="Wytham", odir=odir, trees_odir=trees_odir)

    print("Tiling testing area with no overlap")

    tile_area(test_pc, x_n=2, y_n=11, plot_name="Wytham", odir=odir, trees_odir=trees_odir, overlap=0)




def main():
    # DATA_DIR = "/home/wcherlet/data/Wytham_cleaned/"
    DATA_DIR = "/media/wcherlet/SSD WOUT/backup_IMPORTANT/Wytham_cleaned"
    trees_folder = os.path.join(DATA_DIR, "trees")

    # TILE_DIR = "/home/wcherlet/data/Wytham_cleaned/seperated"
    TILE_DIR = os.path.join(DATA_DIR, "seperated")

    TREES_DIR = os.path.join(TILE_DIR, "trees_kd")
    UNDERSTORY_DIR = os.path.join(TILE_DIR, "understory_kd")

    tree_tiles_dict = read_pointclouds(TREES_DIR)
    understory_tiles_dict = read_pointclouds(UNDERSTORY_DIR)

    # odir = os.path.join(DATA_DIR, "Wytham_train_split")
    odir = "/home/wcherlet/data/Wytham_train_split"

    if not os.path.exists(odir):
        os.makedirs(odir)

    # test_train_split(tree_tiles_dict, understory_tiles_dict, trees_folder, odir)

    tile_wytham(odir)


    return


if __name__ == "__main__":
    main()