import os
import argparse
import glob

import open3d as o3d

def voxel_downsample(folder, voxel_size=0.01, odir=None):
    if not os.path.exists(folder):
        print(f"Cant find folder {folder}")
        os._exit(1)
    if odir is not None and not os.path.exists(odir):
        os.makedirs(odir)

    print(f"Downsampling using voxel down sample to resolution {voxel_size}")

    if os.path.isfile(folder):
        print(f"Processing file {folder}")

        o3dpc = o3d.t.io.read_point_cloud(folder)
        ds_pc  = o3dpc.voxel_down_sample(voxel_size=voxel_size)

        fname = os.path.basename(folder)[:-4] + f"_ds"
        if odir is not None:
            ofile = os.path.join(odir, fname + ".ply")
        else:
            ofile = os.path.join(os.path.dirname(folder), fname+".ply")
        o3d.t.io.write_point_cloud(ofile, ds_pc)

    else:
        print(f"Processing directory {folder}")

        for file in glob.glob(os.path.join(folder, "**", "*.ply"), recursive=True):
            print(file)

            o3dpc = o3d.t.io.read_point_cloud(file)
            ds_pc  = o3dpc.voxel_down_sample(voxel_size=voxel_size)

            fname = os.path.basename(file)[:-4] + '_ds'
            if odir is not None:
                ofile = os.path.join(odir, fname + ".ply")
            else:
                ofile = os.path.join(os.path.dirname(file), fname+".ply")
            o3d.t.io.write_point_cloud(ofile, ds_pc)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, type=str)
    parser.add_argument('-v', "--voxel_size", type=float, default=0.01)
    parser.add_argument('-o', "--odir", type=str, default=None)
    args = parser.parse_args()

    voxel_downsample(args.folder, args.voxel_size, args.odir)

if __name__ == "__main__":
    main()