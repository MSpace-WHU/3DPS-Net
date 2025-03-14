import os
import glob
import random
import time
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from models.kpconv.helper_ply import read_ply, write_ply
from scipy.stats import mode


def plydata_acquire(path):
    # data: [x, y, z, r, g, b, sem_label, ins_label]
    data = read_ply(path)
    x, y, z = data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)

    if 'r' in data.dtype.names:
        r, g, b = data['r'].astype(np.float32), data['g'].astype(np.float32), data['b'].astype(np.float32)
    else:
        r, g, b = data['red'].astype(np.float32), data['green'].astype(np.float32), data['blue'].astype(np.float32)

    if 'treeID' in data.dtype.names:
        # sem_label = data['semantic_seg'].astype(np.float32)
        ins_label = data['treeID'].astype(np.float32)
        return np.column_stack((x, y, z, r, g, b, ins_label))

    if 'instance_label' in data.dtype.names:
        ins_label = data['instance_label'].astype(np.float32)
        return np.column_stack((x, y, z, r, g, b, ins_label))

    if 'class' in data.dtype.names:
        ins_label = data['class'].astype(np.float32)
        return np.column_stack((x, y, z, r, g, b, ins_label))


# check position:(average_x,average_y,average_z) to '.txt' file
def check_position(data_folder, output_folder):
    for file in os.listdir(data_folder):
        output_file = os.path.join(output_folder, 'position_' + file.split('/')[-1][:-4] + '.txt')
        input_file = os.path.join(data_folder, file)
        data = plydata_acquire(input_file)
        with open(output_file, 'w') as f:
            for j in range(int(np.max(data[:, 6]))):
                tree = data[data[:, 6] == j]

                center = np.mean(tree, axis=0)
                # center = (np.max(tree, axis=0) + np.min(tree, axis=0)) / 2.0
                # center = np.mean(tree[tree[:, 2] < (np.min(tree[:, 2]) + np.max(tree[:, 2])) / 2], axis=0)
                f.write('{} {} {} {} {}\n'.format(j, center[0], center[1], center[2], len(tree)))
        f.close
        print('{} is finish'.format(output_file))


# split forest into tree '.ply'
def separation(data_file, output_folder, output_position):
    os.makedirs(output_folder) if not os.path.exists(output_folder) else None
    data = plydata_acquire(data_file)
    position = []
    height = []
    for i in np.unique(data[:, -1]):
        tree = data[data[:, -1] == i]
        # tree_position = np.where(data[:, 6] == i)[0]
        output = os.path.join(output_folder, data_file.split('/')[-1][:-4] + str(i) + '.ply')
        position.append(np.average(tree, axis=0))
        height.append((np.max(tree, axis=0) + np.min(tree, axis=0))[2] / 2)
        write_ply(output, [tree], ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])
    if output_position:
        output_position_file = os.path.join(output_folder, data_file.split('/')[-1][:-8] + 'position.ply')
        write_ply(output_position_file, [np.array(position), np.array(height)],
                  ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label', 'height'])


def match(data_file, position_file, txt_file):
    data = plydata_acquire(data_file)
    position = plydata_acquire(position_file)
    search_tree = KDTree(position[:, :2])
    with open(txt_file, 'w') as f:
        f.write('{}\t{}\t{}\t{}\n'.format('tree_id', 'match_position_id', 'tree_position', 'tree_height'))
        for i in np.unique(data[:, -1]):
            tree = data[data[:, -1] == i]
            tree_center = np.average(tree, axis=0)[:2]
            dis, indices = search_tree.query([tree_center], k=1)
            match_position_id = indices[0][0]
            tree_position = position[match_position_id, :3]
            tree_height = (np.max(tree, axis=0) - np.min(tree, axis=0))[2]
            f.write('{}\t{}\t{}\t{}\n'.format(i, tree_position, match_position_id, tree_height))
    f.close


def merge(data_folder, output_folder, filename):
    os.makedirs(output_folder) if not os.path.exists(output_folder) else None
    merge_data = []
    merge_label = []
    merge_rgb = []
    for file in glob.glob(os.path.join(data_folder, filename + '*')):
        if file[-10:] == '_merge.ply':
            continue

        data = read_ply(file)

        xyz = np.column_stack(
            (data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)))
        merge_data.append(xyz)

        rgb = np.ones_like(xyz) * (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        merge_rgb.append(rgb)

        ins_label = np.ones(len(data)) * int(file.split('/')[-1].split('_')[-1][:-4])
        merge_label.append(ins_label)

    merge_data = np.vstack(merge_data).astype(np.float32)
    merge_rgb = np.vstack(merge_rgb).astype(np.uint8)
    _, merge_label = np.unique(np.concatenate(merge_label), return_inverse=True)
    merge_label = merge_label.astype(np.uint8)

    output_merge_file = os.path.join(output_folder, filename + '_merge.ply')
    write_ply(output_merge_file, [merge_data, merge_rgb, merge_label],
              ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])


# project labels from sampled data into original data '.ply'
def recovery(sampled_file, recovery_file, output_folder, ground_threshold=0.25):
    os.makedirs(output_folder) if not os.path.exists(output_folder) else None
    output_file = os.path.join(output_folder, sampled_file.split('/')[-1][:-4] + '_recovery.ply')
    if os.path.exists(output_file):
        print('{} exists'.format(output_file))
        return 0
    if os.path.exists(sampled_file):
        sampled_data = plydata_acquire(sampled_file)
        original_data = plydata_acquire(recovery_file)

    # sampled_data=sampled_data[sampled_data[:, 6] != 0]
    search_tree = KDTree(sampled_data[:, :3])
    distance, indices = search_tree.query(original_data[:, :3], k=1)

    match_rgb = sampled_data[np.concatenate(indices)][:, 3:6].astype(np.uint8)
    match_label = sampled_data[np.concatenate(indices)][:, -1]

    match_rgb[original_data[:, 2] < ground_threshold] = 255
    match_label[original_data[:, 2] < ground_threshold] = 0

    write_ply(output_file, [original_data[:, :3], match_rgb, match_label],
              ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])


def IOU(pre_file, truth_file, txt_file):
    pre_data = plydata_acquire(pre_file)
    pre_label = pre_data[:, -1]
    truth_data = plydata_acquire(truth_file)
    truth_label = truth_data[:, -1]

    with open(txt_file, 'w') as f:
        for i in range(len(np.unique(pre_label))):
            tree = np.where(pre_label == i)[0]

            if len(tree) == 0:
                continue

            match_label = np.unique(truth_label[tree], return_counts=True)
            if match_label[0][np.argmax(match_label[1])] != 0:
                label = np.argsort(-match_label[1])[0]
            else:
                if len(match_label[1]) > 1:
                    label = np.argsort(-match_label[1])[1]
                else:
                    label = 0

            intersection = match_label[1][label]
            union = len(pre_label[pre_label == i]) + len(
                truth_label[truth_label == match_label[0][label]]) - intersection
            Iou = intersection / union
            f.write('{} {} {}\n'.format(i + 1, int(match_label[0][label]), Iou))
    f.close

    print('{} finished'.format(txt_file))


def reorder(input, output):
    for file in os.listdir(input):
        input_file = os.path.join(input, file)
        output_file = os.path.join(output, file)
        data = read_ply(input_file)
        position_data = np.column_stack(
            (data['x'], data['y'], data['z'], data['r'], data['g'], data['b'], data['instance_label'], data['height']))
        reorder_data = position_data[np.argsort(position_data[:, -1])[::-1]]
        write_ply(output_file, [reorder_data],
                  ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label', 'height'])


if __name__ == '__main__':
    # check_position('/home/user/Desktop/3DPSNet/params/forestsemantic/out',
    # '/home/user/Desktop/3DPSNet/params/forestsemantic')
    # separation('/home/user/Desktop/3DPSNet/data/forestsemantic/labeled_0.02/plot1_0.02.ply',
    #            '/home/user/Desktop/3DPSNet/data/forestsemantic/tree')
    # merge('/home/user/Desktop/3DPSNet/params/forestsemantic/output',
    #       '/home/user/Desktop/3DPSNet/params/forestsemantic/ou', 'plot1')
    for file in glob.glob(os.path.join('/home/user/Desktop/3DPSNet/params/forestsemantic/ou', '*_merge.ply')):
        truth_data = '/home/user/Desktop/3DPSNet/data/forestsemantic/labeled_0.02/plot1_0.02.ply'
        # truth_data = os.path.join('/media/user/data5/000_xty/Forinstance/Forinstance_0.25',
        #                           file.split('/')[-1][:-30] + '_annotated_test_0.25.ply')
        # truth_data = os.path.join('/media/user/data5/000_xty/Forinstance/Forinstance_0.02',
        #                           file.split('/')[-1][:-25] + '_test_test_0.02.ply')
        output = '/home/user/Desktop/3DPSNet/params/output_height2'
        recovery(file, truth_data, output)
        IOU(os.path.join(output, file.split('/')[-1][:-4]+ '_recovery.ply'), truth_data
            , os.path.join(output, file.split('/')[-1][:-4] + '.txt'))
        # IOU(os.path.join(output, file.split('/')[-1][:-30] + '_annotated_test_0.25_recovery.ply'), truth_data
        #     , os.path.join(output, file.split('/')[-1][:-30] + '.txt'))
        # IOU(os.path.join(output, file.split('/')[-1][:-25] + '_test_test_0.02_recovery.ply'), truth_data
        #     , os.path.join(output, file.split('/')[-1][:-25] + '.txt'))
        # match('/home/user/Desktop/3DPSNet/data/forestsemantic/labeled_0.02/plot5_0.02.ply',
        #       '/home/user/Desktop/pointSAM/PointSAM2/params/position/plot5_position.ply',
        #       'plot5_attribute.txt')
        # path = '/home/user/Desktop/3DPSNet/data/forinstance6/test'
        # output_path = '/home/user/Desktop/3DPSNet/data/position_center'
        # for file in os.listdir(path):
        #     path_file = os.path.join(path, file)
        #     separation(path_file, output_path, True)
        # reorder('/home/user/Desktop/3DPSNet/data/position', '/home/user/Desktop/3DPSNet/data/position_reorder')
    # with open('height.txt', 'w') as f:
    #     for file in os.listdir('/home/user/Desktop/3DPSNet/data/position_reorder'):
    #         file_path=os.path.join('/home/user/Desktop/3DPSNet/data/position_reorder',file)
    #         f.write('{}\n'.format(file_path))
    #         data=read_ply(file_path)
    #         height=data['height']
    #         f.write('{} {}\n'.format(len(height[height>height[1]/3]),len(height[height<=height[1]/3])))
    # f.close

