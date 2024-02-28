import os
import sys
import glob

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_pc, merge_pointclouds

def viz_trees_on_tile(tile_file, trees_folder):
    trees = []
    for file in glob.glob(os.path.join(trees_folder, "*.ply")):
        tree = read_pc(file)
        color = np.random.choice(range(256), size=3)/ 256
        legacy_tree = tree.to_legacy()
        legacy_tree.paint_uniform_color(color)
        trees.append(legacy_tree)

    tile = read_pc(tile_file)
    legacy_tile = tile.to_legacy()
    trees.append(legacy_tile)

    o3d.visualization.draw_geometries([legacy_tile])
    o3d.visualization.draw_geometries(trees)

    return

def read_tiles(folder):
    pcs = []
    for file in os.listdir(folder):
        if file[:4] == "tile":
            pc = read_pc(file)
            
            # TODO: TEMP downsample for faster visualization
            pc = pc.voxel_down_sample(voxel_size=0.20)

            legacy_pc = pc.to_legacy()
            bbox = legacy_pc.get_axis_aligned_bounding_box()
            pcs.append(legacy_pc)
            # pcs.append(bbox)
    return pcs

def compare_tiles(tile_file1, tile_file2):

    tile1 = read_pc(tile_file1)
    tile2 = read_pc(tile_file2)

    # move next to each other
    x_loc = tile1.get_min_bound().numpy()[0]
    y_loc = tile1.get_max_bound().numpy()[1] + 5
    z_loc = tile1.get_min_bound().numpy()[2]
    translation = o3d.core.Tensor([x_loc, y_loc, z_loc]) - tile2.get_min_bound()
    tile2 = tile2.translate(translation)

    tile1_leg = tile1.to_legacy()
    tile2_leg = tile2.to_legacy()

    print(len(tile1_leg.points))
    print(len(tile2_leg.points))
    o3d.visualization.draw_geometries([tile1_leg, tile2_leg])

    return


def litchfield_full_plot(understory_tiles, trees_parent_folder, august=True):

    pcs = read_tiles(understory_tiles)
    merged_understory = merge_pointclouds(pcs)

    max_bound = merged_understory.get_max_bound()
    min_bound = merged_understory.get_min_bound()

    print(f"Max_bound of plot: {max_bound}")
    print(f"Min_bound of plot: {min_bound}")
    print(f"Dimension of plot: {max_bound-min_bound}")

    o3d.visualization.draw_geometries(pcs)

    return


def get_xy_view(understory_tiles):
    tilenames = [f for f in sorted(os.listdir(understory_tiles)) if f[-3:] == 'txt']
    
    x = []
    y = []
    for tilename in tilenames:
        file = os.path.join(understory_tiles, tilename)
        pc = read_pc(file)

        center = (pc.get_max_bound().numpy() + pc.get_min_bound().numpy())/2
        x.append(center[0])
        y.append(center[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(tilenames):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()


def main():

    ### COMPARE UNDERSTORY TILES

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_SEPT"
    # TILE_FILE1 = os.path.join(FOLDER, "tile_0_-20_SEP_US_OK.txt")

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    # TILE_FILE2 = os.path.join(FOLDER, "tile_0_-20_AUG_US_OK.txt")

    # compare_tiles(TILE_FILE1, TILE_FILE2)

    # -----------------------------------

    ### COMPARE FULL TILES

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_SEPT"
    # TILE_FILE1 = os.path.join(FOLDER, "tile_0_-20_SEP_US_OK.txt")

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    # TILE_FILE2 = os.path.join(FOLDER, "tile_0_-20_AUG_US_OK.txt")

    # compare_tiles(TILE_FILE1, TILE_FILE2)

    # -----------------------------------

    ### VISUALIZE TREES ON TILE


    FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    TILE_FILE = os.path.join(FOLDER, "tile_40_-60_AUG_US_OK.txt")
    PC_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Bomen/tile_40_-60_BOMEN/Augustus"


    viz_trees_on_tile(TILE_FILE, PC_FOLDER)

    # -----------------------------------

    ### VISUALIZE FULL PLOT

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    # PC_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Bomen/"
    # AUGUST = TRUE

    # litchfield_full_plot(FOLDER, PC_FOLDER, august=AUGUST)

    return

if __name__ == "__main__":
    main()