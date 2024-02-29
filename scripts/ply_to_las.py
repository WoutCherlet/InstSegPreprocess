import os
import argparse
import glob

import numpy as np
import open3d as o3d
import laspy


def ply_to_las(folder, odir=None):

    if not os.path.exists(folder):
        print(f"Cant find folder {folder}")
        os._exit(1)
    if odir is not None and not os.path.exists(odir):
        os.makedirs(odir)
    
    if os.path.isfile(folder):
        print(f"Processing file {folder}")

        o3dpc = o3d.t.io.read_point_cloud(folder)
        points = o3dpc.point.positions.numpy()

        fname = os.path.basename(folder)[:-4]
        if odir is not None:
            ofile = os.path.join(odir, fname + ".las")
        else:
            ofile = os.path.join(os.path.dirname(folder), fname+".las")

        header = laspy.LasHeader(version="1.4", point_format=0)
        header.scales = np.array([0.000001, 0.000001, 0.000001])
        las = laspy.LasData(header)
        las.x = points[:,0]
        las.y = points[:,1]
        las.z = points[:,2]
        las.write(ofile)
    else:
        print(f"Processing directory {folder}")

        for file in glob.glob(os.path.join(folder, "**", "*.ply"), recursive=True):
            print(file)

            o3dpc = o3d.t.io.read_point_cloud(file)
            points = o3dpc.point.positions.numpy()

            fname = os.path.basename(file)[:-4]
            if odir is not None:
                ofile = os.path.join(odir, fname + ".las")
            else:
                ofile = os.path.join(os.path.dirname(file), fname+".las")

            header = laspy.LasHeader(version="1.4", point_format=0)
            header.scales = np.array([0.000001, 0.000001, 0.000001])
            las = laspy.LasData(header)
            las.x = points[:,0]
            las.y = points[:,1]
            las.z = points[:,2]
            las.write(ofile)

    return

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, type=str)
    parser.add_argument('-o', "--odir", type=str, default=None)
    args = parser.parse_args()

    ply_to_las(args.folder, args.odir)

if __name__ == "__main__":
    main()