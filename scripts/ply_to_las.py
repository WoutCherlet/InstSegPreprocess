import os
import argparse
import glob

import numpy as np
import open3d as o3d
import laspy


def ply_to_las():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, type=str)
    parser.add_argument('-o', "--odir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"Cant find folder {args.folder}")
        os._exit(1)

    if args.odir is not None and not os.path.exists(args.odir):
        os.makedirs(args.odir)
    
    if os.path.isfile(args.folder):
        print(f"Processing file {args.folder}")

        o3dpc = o3d.t.io.read_point_cloud(args.folder)
        points = o3dpc.point.positions.numpy()

        fname = os.path.basename(args.folder)[:-4]

        if args.odir is not None:
            ofile = os.path.join(args.odir, fname + ".las")
        else:
            ofile = os.path.join(os.path.dirname(args.folder), fname+".las")

        header = laspy.LasHeader(version="1.4", point_format=0)
        header.scales = np.array([0.000001, 0.000001, 0.000001])
        las = laspy.LasData(header)
        las.x = points[:,0]
        las.y = points[:,1]
        las.z = points[:,2]
        las.write(ofile)
    else:
        print(f"Processing directory {args.folder}")

        for file in glob.glob(os.path.join(args.folder, "**", "*.ply"), recursive=True):
            print(file)

            o3dpc = o3d.t.io.read_point_cloud(file)
            points = o3dpc.point.positions.numpy()

            fname = os.path.basename(file)[:-4]
            if args.odir is not None:
                ofile = os.path.join(args.odir, fname + ".las")
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
            

if __name__ == "__main__":
    ply_to_las()