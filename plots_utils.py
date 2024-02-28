import os
import open3d as o3d
import numpy as np

from tree_io import read_ply_folder

def trees_in_plot(plot, trees, odir, threshold = 0.9, output_all=True):

    # function to detect if tree is entirely in plot
    
    odir_in_plot = os.path.join(odir, f"in_plot_th{threshold:.2f}")
    odir_out_plot = os.path.join(odir, f"out_plot_th{threshold:.2f}")

    if not os.path.exists(odir_in_plot):
        os.makedirs(odir_in_plot)
    if not os.path.exists(odir_out_plot):
        os.makedirs(odir_out_plot)


    # get bbox of plot (assumes plot is rectangular and axis-aligned)
    max_bound = plot.get_max_bound().numpy()
    min_bound = plot.get_min_bound().numpy()

    plot_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = min_bound, max_bound = max_bound)

    # divide trees into inside and outside plot based on threshold value
    for file in trees:
        pc = trees[file]
        in_idx = plot_bbox.get_point_indices_within_bounding_box(pc.point.positions)

        n_in = in_idx.numpy().shape[0]
        in_prop =  n_in / len(pc.point.positions.numpy())

        if in_prop >= threshold:
            out_path = os.path.join(odir_in_plot, file)
        else:
            out_path = os.path.join(odir_out_plot, file)

        if n_in > 0 and output_all:
            o3d.t.io.write_point_cloud(odir, file)

        
        o3d.t.io.write_point_cloud(out_path, pc)
    
    return

def tile_area(merged_area, x_n, y_n, plot_name, odir, trees_odir=None, overlap=5):

    min_bound = merged_area.get_min_bound().numpy()
    max_bound = merged_area.get_max_bound().numpy()

    x_tile_size = (max_bound[0] - min_bound[0] - overlap) / x_n + overlap
    y_tile_size = (max_bound[1] - min_bound[1] - overlap) / y_n + overlap

    print(f"Tile sizes: {x_tile_size}, {y_tile_size}")

    if trees_odir is not None:
        trees = read_ply_folder(trees_odir)

    
    if overlap != 5:
        odir = os.path.join(odir, f"overlap_{overlap}")
        trees_odir = os.path.join(trees_odir, f"overlap_{overlap}")

    tile_n = 0

    all_tiles = []
    
    for i in range(x_n):
        for j in range(y_n):
            tile_min_bound = min_bound.copy()
            tile_min_bound[0] += i*(x_tile_size - overlap)
            tile_min_bound[1] += j*(y_tile_size - overlap)

            tile_max_bound = max_bound.copy()
            tile_max_bound[0] = tile_min_bound[0] + x_tile_size
            tile_max_bound[1] = tile_min_bound[1] + y_tile_size

            tile_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = tile_min_bound, max_bound = tile_max_bound)
            tile_pc = merged_area.crop(tile_bbox)

            cur_tile = f"{plot_name}_{tile_n}"

            o3d.t.io.write_point_cloud(os.path.join(odir, f"{cur_tile}.ply"), tile_pc)
            tile_n += 1

            # extra: keep track of trees on tile
            if trees_odir is not None:
                tile_trees_odir = os.path.join(trees_odir, cur_tile)

                trees_in_plot(tile_pc, trees, tile_trees_odir, threshold=0.9, output_all=True)

            # TODO: temp: shift and save
            # tile_pc = tile_pc.translate(np.array([i*7, j*7, 0]))
            # all_tiles.append(tile_pc)

    # TODO: TEMP
    # o3d.visualization.draw_geometries(all_tiles)
    pass



if __name__ == "__main__":
    plot = "/home/wcherlet/BenchmarkPaper/data/BASE/test_merged.ply"
    tree_dir = "/home/wcherlet/BenchmarkPaper/data/BASE/trees/test/"
    odir = "/home/wcherlet/BenchmarkPaper/data/BASE/test_trees_thresholded"

    plot = o3d.t.io.read_point_cloud(plot)
    trees = read_ply_folder(tree_dir)
    trees_in_plot(plot, trees, odir, threshold=0.9, output_all=True)