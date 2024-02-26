import os
import sys

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tree_io import read_txt, read_las

def viz_pc(pc):
    bbox = pc.to_legacy().get_axis_aligned_bounding_box()
    o3d.visualization.draw_geometries([pc.to_legacy(), bbox])
    return

def merge_all(pointclouds):
    pointcloud = pointclouds[0]
    for pc in pointclouds[1:]:
        pointcloud += pc
    return pointcloud

def viz_trees_on_tile(tile_file, trees_folder):

    trees = []

    for file in os.listdir(trees_folder):

        if file[-4:] == ".ply":
            tree = o3d.t.io.read_point_cloud(os.path.join(trees_folder, file))
        # tree = read_txt(os.path.join(trees_folder, file))

        color = np.random.choice(range(256), size=3)/ 256

        legacy_tree = tree.to_legacy()
        legacy_tree.paint_uniform_color(color)

        trees.append(legacy_tree)

    if tile_file[-4:] == ".txt":
        tile = read_txt(tile_file)
    elif tile_file[-4:] == ".las":
        tile = read_las(tile_file)
    else:
        print(f"Cant read tile {tile_file}")
        return
    
    legacy_tile = tile.to_legacy()

    trees.append(legacy_tile)

    o3d.visualization.draw_geometries([legacy_tile])

    o3d.visualization.draw_geometries(trees)

def read_tiles(folder, extension=".txt"):
    pcs = []

    for file in os.listdir(folder):
        if file[:4] == "tile":
            real_extension = file[-4:]
            if extension == ".txt" and real_extension == extension:
                pc = read_txt(os.path.join(folder, file))
            elif extension == ".las" and real_extension == extension:
                pc = read_las(os.path.join(folder, file))
            elif extension != real_extension:
                print(f"Extension {real_extension} doesnt match given extension {extension}")
                continue
            else:
                print(f"Unable to read file {file}")
                continue
            
            # TODO: TEMP downsample for faster visualization
            pc = pc.voxel_down_sample(voxel_size=0.20)

            legacy_pc = pc.to_legacy()
            bbox = legacy_pc.get_axis_aligned_bounding_box()

            pcs.append(legacy_pc)
            # pcs.append(bbox)

    return pcs

def compare_tiles(tile_file1, tile_file2):

    tile1 = read_txt(tile_file1)

    tile2 = read_txt(tile_file2)

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

    




def litchfield_full_plot(understory_tiles, trees_parent_folder):

    pcs = read_tiles(understory_tiles)

    merged_understory = merge_all(pcs)

    max_bound = merged_understory.get_max_bound()
    min_bound = merged_understory.get_min_bound()

    print(f"Max_bound of plot: {max_bound}")
    print(f"Min_bound of plot: {min_bound}")
    print(f"Dimension of plot: {max_bound-min_bound}")

    o3d.visualization.draw_geometries(pcs)


def get_xy_view(understory_tiles):
    tilenames = [f for f in sorted(os.listdir(understory_tiles)) if f[-3:] == 'txt']
    
    x = []
    y = []
    for tilename in tilenames:
        file = os.path.join(understory_tiles, tilename)

        pc = read_txt(file)

        center = (pc.get_max_bound().numpy() + pc.get_min_bound().numpy())/2

        x.append(center[0])
        y.append(center[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(tilenames):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()


    

def main():

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_SEPT"
    # TILE_FILE1 = os.path.join(FOLDER, "tile_0_-20_SEP_US_OK.txt")


    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    # TILE_FILE2 = os.path.join(FOLDER, "tile_0_-20_AUG_US_OK.txt")

    # compare_tiles(TILE_FILE1, TILE_FILE2)

    # FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/TILES_litchfield_sep_all36comb_AOI_dev50_refl-15_1cm/"
    # TILE_FILE = os.path.join(FOLDER, "tile_0_-20.las")


    FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Understorey/OK_TILES_AUG"
    TILE_FILE = os.path.join(FOLDER, "tile_0_-20_AUG_US_OK.txt")
    PC_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Bomen/tile_0_-20_BOMEN/Augustus"

    viz_trees_on_tile(TILE_FILE, PC_FOLDER)

    # get_xy_view(FOLDER)
    # litchfield_full_plot(FOLDER, PC_FOLDER)

    return

if __name__ == "__main__":
    main()