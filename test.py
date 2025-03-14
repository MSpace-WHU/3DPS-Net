import os
import glob
import random

import numpy as np
import torch
import argparse
import open3d as o3d
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from models.dataset import ForestDataset_val, ForestDataset_val_collate_fn
from models.network import PSNet3d
from functools import partial
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from models.utils import to_o3d_pcd, yellow, blue, gray, processbar, Logger, farthest_point_sample
from models.kpconv.helper_ply import write_ply, read_ply
import pdb


def select_plot(data, position, grid_size=1):
    x_min = position[0] - grid_size / 2
    y_min = position[1] - grid_size / 2
    x_max = x_min + grid_size
    y_max = y_min + grid_size
    return data[(data[:, 0] >= x_min) & (data[:, 0] <= x_max) & (data[:, 1] >= y_min) & (data[:, 1] <= y_max)]


def data_acquire(path, flag=[0, 1, 2, 3, 4, 5, 6, 7]):
    if path[-4:] == '.txt':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        return np.loadtxt(path, dtype=np.float32)
    elif path[-4:] == '.ply':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        data = read_ply(path)
        id = data.dtype.names
        x = data[id[flag[0]]].astype(np.float32)
        y = data[id[flag[1]]].astype(np.float32)
        z = data[id[flag[2]]].astype(np.float32)
        sem_label = data[id[flag[6]]].astype(np.float32)
        ins_label = data[id[flag[7]]].astype(np.float32)
        return np.column_stack((x, y, z, np.ones_like(x), np.ones_like(x), np.ones_like(x), sem_label, ins_label))


# def plydata_acquire(path):
#     # data: [x, y, z, r, g, b, sem_label, ins_label]
#     data = read_ply(path)
#     x, y, z = data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)
#     r, g, b = data['r'].astype(np.float32), data['g'].astype(np.float32), data['b'].astype(np.float32)
#     if 'semantic_seg' in data.dtype.names:
#         sem_label = data['semantic_seg'].astype(np.float32)
#         ins_label = data['treeID'].astype(np.float32)
#         return np.column_stack((x, y, z, r, g, b, sem_label, ins_label))
#
#     if 'instance_label' in data.dtype.names:
#         ins_label = data['instance_label'].astype(np.float32)
#         return np.column_stack((x, y, z, r, g, b, ins_label))

def IOU(tree, data, truth_data):
    # accurary
    _, indices = tree.query(data, k=1)
    match_data = truth_data[np.concatenate(indices)]
    match_label = np.unique(match_data[:, 7], return_counts=True)

    intersection_count = np.max(match_label[1])
    # intersection = match_data[match_data[:, 7] == match_label]

    match_label = int(match_label[0][np.argmax(match_label[1])])
    match_label_tree = truth_data[truth_data[:, 7] == match_label]
    union = np.unique(np.concatenate((match_label_tree[:, :3], data), axis=0), axis=0)

    Iou = intersection_count / len(union)
    return match_label, Iou


def pick_points_from_cloud(pcd, required_points=3):
    picked_indices = []
    while len(picked_indices) < required_points:
        print(f"please select {required_points} points。")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        temp_picked = vis.get_picked_points()
        if temp_picked:
            picked_indices.extend(temp_picked)
        if len(picked_indices) >= required_points:
            break
        print("Not enough")
    return picked_indices[:required_points]


def visualize(prediction_data, label, prompt_ind):
    pcd = to_o3d_pcd(prediction_data, blue)
    np.asarray(pcd.colors)[label == 1] = np.array(yellow)
    np.asarray(pcd.colors)[prompt_ind] = np.array([1, 0, 0])
    o3d.visualization.draw_geometries([pcd], width=1000, height=800)


def merge_individual_tree(output_path, filename, search_tree, truth_data, logger):
    merge_data = []
    merge_label = []
    merge_rgb = []
    for file in glob.glob(os.path.join(output_path, filename + '*')):
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
    # print(np.unique(merge_label))

    _, indices = search_tree.query(merge_data, k=1)
    indices = np.squeeze(indices, axis=1)
    merge_data_output = np.hstack(
        (truth_data[:, :3], np.ones((truth_data.shape[0], 3)) * 255, np.zeros((truth_data.shape[0], 1))))
    merge_data_output[indices] = np.concatenate((merge_data, merge_rgb, merge_label[:, np.newaxis]), axis=1)
    output_merge = os.path.join(output_path, filename + '_merge.ply')
    write_ply(output_merge, [merge_data_output[:, :3], merge_data_output[:, 3:6].astype(np.uint8),
                             merge_data_output[:, 6].astype(np.uint8)],
              ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])

    logger.write('Match label:{}'.format(len(truth_label)))
    logger.write('{}'.format(sorted(truth_label)))
    logger.write('Groud Truth:{}'.format(len(np.unique(truth_data[:, 7]))-1))
    logger.write('{}'.format(np.mean(Mean_Iou)))


if __name__ == '__main__':

    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./test.py --pretrained=[pretrain_path]

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--dataset_name', default='forinstance1', type=str, help='The name of folder')
    # parser.add_argument('--mode', type=str, default='trainval', help='name')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model path')
    parser.add_argument('--position', default=None, type=str, help='position path')
    parser.add_argument('--output', default=None, type=str, help='output')
    parser.add_argument('--option', default='prompt', type=str, help='prompt,auto,mix')
    parser.add_argument('--count_num', default=1000, type=int, help='iterations')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    device = torch.device(args.local_rank)

    # Configuration
    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_dir, 'params', args.dataset_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    logger = Logger(args.dataset_name, os.path.join(save_path, 'log_val.txt'))
    output_path = args.output if args.output else os.path.join(save_path, 'output')
    os.makedirs(output_path) if not os.path.exists(output_path) else None

    # Dataset_configuration
    test_dataset = ForestDataset_val(root_dir, args.dataset_name, split='test')

    test_sampler = DistributedSampler(test_dataset,
                                      num_replicas=dist.get_world_size(),
                                      rank=args.local_rank,
                                      shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=8,
                             collate_fn=partial(ForestDataset_val_collate_fn))

    # network_configuration
    network = PSNet3d().cuda(args.local_rank)
    network = DistributedDataParallel(network,
                                      device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)

    if args.pretrained:
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(args.pretrained, map_location='cpu').items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)
    else:
        raise ValueError("No Model")

    network.eval()

    # Hyperparameters
    count_num = args.count_num
    truth_threshold = 0.1

    with (torch.no_grad()):

        for batch_idx, (pcd_list, label_list) in enumerate(test_loader):

            points = pcd_list[0][:, 0:3]
            mask = np.zeros(len(points))

            truth_data = test_loader.dataset.data[batch_idx]
            threshold = np.min(truth_data[:, 2]) + 4
            print(threshold)
            search_tree = KDTree(truth_data[:, :3])

            truth_label = []
            Mean_Iou = []

            data_file = test_loader.dataset.datapath[batch_idx]
            filename = data_file.split('/')[-1][:-4]
            logger.write('{}'.format(filename))
            # mode++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if args.option == 'auto' or args.option == 'mix':
                step_count = 0
                step_height = 0
                for i in range(count_num):
                    if mask.any():
                        points = points[mask]
                        mask = np.zeros(len(points))
                        # if points[points[:, 2] > threshold - 3].mean(axis=0)[-1] < threshold:
                        #     break

                    picked_points = torch.tensor([0.0, 0.0, 0.0])
                    while picked_points[-1] < threshold:
                        prompt_ind = np.random.choice(range(len(points)))
                        picked_points = points[prompt_ind]
                        print(picked_points)

                    select_data = select_plot(points, picked_points, 8.5)
                    prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points.to('cpu').numpy())[0][0]
                    select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))

                    feats = network([select_data], [prompt_ind], device)

                    prediction_data = np.array(select_data[:, :3])
                    label = np.concatenate((feats[0] - 0.4).round().to('cpu').numpy())

                    # visualize(prediction_data, label, prompt_ind)

                    output = os.path.join(output_path, filename + '_' + str(i + 1) + '.ply')
                    output_data = prediction_data[label == 1]

                    if i % 300 == 0:
                        step_count = step_count + 100
                        step_height = step_height + 1
                    if len(output_data) < 800 - step_count or \
                            np.average(output_data, axis=0)[-1] < threshold - step_height:
                        continue

                    # accurary
                    match_label, Iou = IOU(search_tree, output_data, truth_data)
                    truth_label.append(match_label)
                    Mean_Iou.append(Iou)

                    write_ply(output, [output_data], ['x', 'y', 'z'])
                    # position_output = np.mean(output_data, axis=0)
                    # position_output2 = (np.max(output_data, axis=0) + np.min(output_data, axis=0)) * 0.5
                    # print('{} {} {} {}'.format(str(i + 1), position_output[0], position_output[1], position_output[2]))
                    mask = ~np.isin(points.to('cpu').numpy(), output_data).all(axis=1)

            # mode++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if args.option == 'prompt' or args.option == 'mix':
                i = 0
                if args.position:
                    # pdb.set_trace()
                    position_file = glob.glob(os.path.join(args.position,  data_file.split('/')[-1][:5] + '*position.ply'))[0]
                    print(position_file)

                    position_points = data_acquire(position_file)[1:]
                    print(position_points)

                    points = pcd_list[0][:, 0:3]
                    mask = np.zeros_like(points)
                    i = 0
                    for position_point_data in position_points:
                        position_point_data = position_point_data[:3]
                        # print(position_point_data)

                        if mask.any():
                            points = points[mask]
                            mask = np.zeros(len(points))

                        search_tree2 = KDTree(points[:, :3])
                        select_data = select_plot(points, position_point_data, 8.5)
                        position_point_data[2] = 16
                        print(position_point_data)
                        dis, indices = search_tree2.query([position_point_data], k=1)
                        picked_points = np.squeeze(points[indices])
                        print(picked_points)

                        prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points.to('cpu').numpy())[0][0]
                        select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))

                        feats = network([select_data], [prompt_ind], device)

                        prediction_data = np.array(select_data[:, :3])

                        truth_threshold = 0.1
                        label = np.concatenate((feats[0] - truth_threshold + 0.5).round().to('cpu').numpy())

                        output = os.path.join(output_path, filename + '_' + str(i + 1) + '.ply')
                        output_data = prediction_data[label == 1]

                        # accurary
                        match_label, Iou = IOU(search_tree, output_data, truth_data)
                        logger.write('The iou of Tree {} corresponds to Tree {} is: {}'.format(i, match_label, Iou))
                        truth_label.append(match_label)
                        Mean_Iou.append(Iou)

                        write_ply(output, [output_data], ['x', 'y', 'z'])
                        mask = ~np.isin(points.to('cpu').numpy(), prediction_data[label == 1]).all(axis=1)
                        i = i + 1
                else:
                    confirm = input("comfirm?(y(yes)/n(no)):")
                    while confirm != 'y':
                        pcd = to_o3d_pcd(points[points[:, 2] > threshold - 3], blue)
                        prompt_ind = pick_points_from_cloud(pcd, 1)  # id
                        picked_points = np.asarray(pcd.points)[prompt_ind][0]  # xyz
                        select_data = select_plot(points, picked_points, 8)

                        prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points)[0][0]
                        select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))

                        feats = network([select_data], [prompt_ind], device)

                        prediction_data = np.array(select_data[:, :3])
                        label = np.concatenate((feats[0] - truth_threshold + 0.5).round().to('cpu').numpy())
                        output = os.path.join(output_path, filename + '_' + str(i + 1) + '.ply')
                        i = i + 1
                        output_data = prediction_data[label == 1]

                        if len(output_data) >= 800:
                            match_label, Iou = IOU(search_tree, output_data, truth_data)
                            logger.write('The iou of Tree {} corresponds to Tree {} is: {}'.format(i, match_label, Iou))
                            if match_label != 0:
                                truth_label.append(match_label)
                                Mean_Iou.append(Iou)
                            write_ply(output, [output_data], ['x', 'y', 'z'])
                            mask = ~np.isin(points.to('cpu').numpy(), prediction_data[label == 1]).all(axis=1)

                            points = points[mask]

                        pcdcut = to_o3d_pcd(output_data, blue)
                        # np.asarray(pcdcut.colors)[label == 1] = np.array(yellow)
                        # np.asarray(pcdcut.colors)[prompt_ind] = np.array([1, 0, 0])
                        o3d.visualization.draw_geometries([pcdcut], width=1000, height=800, window_name="pcd %d")

                        confirm = input("comfirm?(y(yes)/n(no)):")

            merge_individual_tree(output_path, filename, search_tree, truth_data, logger)
