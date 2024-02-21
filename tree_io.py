#!/usr/bin/env python3

import os
import glob
import open3d as o3d

__all__ = ['read_pointclouds', 'merge_pointclouds', 'down_sample_folder']


def read_pointclouds(folder):
    out = {}

    for file in glob.glob(os.path.join(folder, "*.ply")):
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

def merge_pointclouds(pointclouds):
    pointcloud = pointclouds[0]
    for pc in pointclouds[1:]:
        pointcloud += pc
    return pointcloud


def down_sample_folder(pc_folder):
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']

    odir = os.path.join(pc_folder, "vis_ds")
    if not os.path.exists(odir):
        os.mkdir(odir)

    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(os.path.join(pc_folder, filename))

        pcl = pcl.voxel_down_sample(voxel_size=0.20)

        out_path = os.path.join(odir, filename)

        o3d.io.write_point_cloud(out_path, pcl)