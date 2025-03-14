import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import numpy as np
import os
import glob
import pandas as pd
import pylas
from models.kpconv.helper_ply import read_ply, write_ply

if __name__ == '__main__':
    input = '/home/user/Desktop/ForestSemantic'
    output = '/home/user/Desktop/3DPSNet/data/forest1/test'
    os.makedirs(output) if not os.path.exists(output) else None
    ply_file = glob.glob(os.path.join(input, '*.ply'))
    txt_file = glob.glob(os.path.join(input, '*.txt'))
    las_file = glob.glob(os.path.join(input, '*.las'))
    grid = 0.25

    if ply_file:
        for file in ply_file:
            output_path = os.path.join(output, file.split('/')[-1][:-4] + '_' + str(grid) + '.ply')
            if os.path.exists(output_path):
                continue
            print(output_path)

            data = read_ply(file)
            point = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
            features = np.zeros_like(point)
            sem_labels = data['semantic_seg'].astype(np.uint8)
            ins_labels = data['treeID'].astype(np.uint8)

            sub_xyz, _, sub_semlabel = cpp_subsampling.subsample(point, features=features, classes=sem_labels,
                                                                 sampleDl=grid)
            _, _, sub_inslabel = cpp_subsampling.subsample(point, features=features, classes=ins_labels, sampleDl=grid)

            write_ply(output_path,
                      [sub_xyz, np.zeros_like(sub_xyz), sub_semlabel.astype(np.uint8), sub_inslabel.astype(np.uint8)],
                      ['x', 'y', 'z', 'r', 'g', 'b', 'semantic_seg', 'treeID'])



    if txt_file:
        for file in txt_file:
            output_path = os.path.join(output, file.split('/')[-1][:-4] + '_' + str(grid) + '.ply')
            if os.path.exists(output_path):
                continue
            print(output_path)

            data = pd.read_csv(file, delim_whitespace=True).values
            point = data[:, :3].astype(np.float32)
            features = data[:, 3:6].astype(np.float32)
            sem_labels = data[:, 6].astype(np.uint8)
            ins_labels = data[:, 7].astype(np.uint8)

            sub_xyz, _, sub_semlabel = cpp_subsampling.subsample(point, features=features, classes=sem_labels,
                                                                 sampleDl=grid)
            _, _, sub_inslabel = cpp_subsampling.subsample(point, features=features, classes=ins_labels, sampleDl=grid)

            write_ply(output_path,
                      [sub_xyz, np.zeros_like(sub_xyz), sub_semlabel.astype(np.uint8), sub_inslabel.astype(np.uint8)],
                      ['x', 'y', 'z', 'r', 'g', 'b', 'semantic_seg', 'treeID'])

            # output_data = pd.DataFrame(np.column_stack((sub_xyz, sub_semlabel, sub_inslabel)))
            # output_data.to_csv(output_path, header=None, index=None, sep=" ")

    if las_file:
        for file in las_file:
            output_path = os.path.join(output, file.split('/')[-1][:-4] + '_' + str(grid) + '.ply')
            if os.path.exists(output_path):
                continue
            print(output_path)

            las = pylas.read(file)
            id = las.points.dtype.names
            print(id)
            x = las.points[id[0]] * las.header.x_scale
            y = las.points[id[1]] * las.header.y_scale
            z = las.points[id[2]] * las.header.z_scale
            point = np.stack([x, y, z], axis=1).astype(np.float32)
            features = np.stack([las.points[id[3]], las.points[id[4]], las.points[id[5]]], axis=1).astype(np.float32)
            sem_labels = las.points[id[6]].astype(np.uint8)
            ins_labels = las.points[id[7]].astype(np.uint8)

            sub_xyz, _, sub_semlabel = cpp_subsampling.subsample(point, features=features, classes=sem_labels,
                                                                 sampleDl=grid)
            _, _, sub_inslabel = cpp_subsampling.subsample(point, features=features, classes=ins_labels, sampleDl=grid)

            write_ply(output_path,
                      [sub_xyz, np.zeros_like(sub_xyz), sub_semlabel.astype(np.uint8), sub_inslabel.astype(np.uint8)],
                      ['x', 'y', 'z', 'r', 'g', 'b', 'semantic_seg', 'treeID'])


    print('Finished')
