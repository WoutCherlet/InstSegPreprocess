#!/usr/bin/env python3

import os
import glob
import open3d as o3d
import numpy as np
import laspy

__all__ = ['read_ply_folder', 'merge_pointclouds', 'down_sample_ply_folder', 'read_pc', 'read_txt', 'read_las']


def read_ply_folder(folder):
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


def down_sample_ply_folder(pc_folder):
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']

    odir = os.path.join(pc_folder, "vis_ds")
    if not os.path.exists(odir):
        os.mkdir(odir)

    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(os.path.join(pc_folder, filename))

        pcl = pcl.voxel_down_sample(voxel_size=0.20)

        out_path = os.path.join(odir, filename)

        o3d.io.write_point_cloud(out_path, pcl)

def read_pc(file):
    ext = file[-4:]

    if ext == ".txt":
        return read_txt(file)
    elif ext == ".las":
        return read_las(file)
    elif ext == ".ply":
        return o3d.t.io.read_point_cloud(file)
    else:
        print(f"ERROR: cant read pc {file}")
        return

def read_txt(file):
    arr = np.loadtxt(file, dtype=float, skiprows=1)
    o3d_pc = o3d.t.geometry.PointCloud()
    o3d_pc.point.positions = o3d.core.Tensor(arr[:,:3])
    return o3d_pc

def read_las(file):
    point_cloud = laspy.read(file)
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

    o3d_pc = o3d.t.geometry.PointCloud()
    o3d_pc.point.positions = o3d.core.Tensor(points)
    return o3d_pc