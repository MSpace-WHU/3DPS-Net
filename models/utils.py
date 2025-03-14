import numpy as np
import open3d as o3d
import torch
import logging

blue, yellow, gray = [0, 0.651, 0.929], [1, 0.706, 0], [0.7, 0.7, 0.7]


def processbar(current, totle):
    process_str = ""
    for i in range(int(20 * current / totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|   %d / %d" % (process_str, current, totle)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array([colors] * pts.shape[0]))
    return pcd


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        print("\rfps process: %d / %d" % (i + 1, npoint), end="")
    print()
    return centroids


class Logger(object):

    def __init__(self, name, save_path, log_level_INFO=True,
                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        super(Logger, self).__init__()

        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(level=logging.INFO if log_level_INFO else logging.WARN)  # logging.WARN

        self.handler = logging.FileHandler(save_path)
        self.handler.setLevel(level=logging.INFO if log_level_INFO else logging.WARN)  # logging.WARN
        formatter = logging.Formatter(format)
        self.handler.setFormatter(formatter)

        self.console = logging.StreamHandler()
        self.console.setLevel(level=logging.INFO if log_level_INFO else logging.WARN)  # logging.WARN
        self.console.setFormatter(formatter)

    def write(self, message):
        self.logger.addHandler(self.handler)
        self.logger.addHandler(self.console)

        # if isinstance(message, (dict, list, str)):
        #     message = str(message)  # Convert dictionaries and lists to string
        #     self.logger.info(message)
        # else:
        #     raise ValueError("Message type must be dict, list, or str.")

        assert type(message) in [dict, list, str]
        if type(message) == str:
            self.logger.info(message)
        else:
            raise ValueError("Message type must be str.")

    def write_only(self, message):
        self.logger.addHandler(self.handler)

        assert type(message) in [dict, list, str]
        if type(message) == str:
            self.logger.info(message)
        else:
            raise ValueError("Message type must be str.")
