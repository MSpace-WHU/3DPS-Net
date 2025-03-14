import pdb

import numpy as np
import torch
from torch import nn
from models.kpconv.blocks import block_decider
from models.kpconv.kernel_points import create_3D_rotations
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from models.transformer import SelfAttention, PositionEmbeddingCoordsSine


class Config:
    def __init__(self):
        self.num_layers = 5
        self.in_points_dim = 3
        self.first_feats_dim = 256
        self.first_subsampling_dl = 0.2
        self.in_feats_dim = 4
        self.final_feats_dim = 1
        self.conv_radius = 2.5
        self.deform_radius = 5.0
        self.num_kernel_points = 15
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False
        self.neighborhood_limits = [96, 48, 48, 24, 24]


        self.architecture = ['simple',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb',
                             'nearest_upsample',
                             'unary',
                             'nearest_upsample',
                             'unary',
                             'nearest_upsample',
                             'unary',
                             'nearest_upsample',
                             'unary',
                             'last_unary']


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=False):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    对一批3D点云进行网格子采样。这个函数可以处理仅包含点坐标的点云，也可以处理带有额外特征或标签的点云。
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """
    R = None
    B = len(batches_len)
    # pdb.set_trace()
    if not isinstance(points, np.ndarray):
        points = points.to('cpu').numpy()

    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.numpy().copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    # 如果没有提供特征和标签，只进行点的子采样
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose
        )

        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length

        return torch.from_numpy(s_points), torch.from_numpy(s_len)
    # 如果提供了特征但没有标签，对点和特征进行子采样
    elif labels is None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            features=features,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose
        )

        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length

        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)
    # 如果没有提供特征但提供了标签，对点和标签进行子采样
    elif features is None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            classes=labels,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose
        )

        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length

        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)
    # 如果提供了特征和标签，对点、特征和标签进行子采样
    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            features=features,
            classes=labels,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose
        )

        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length

        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(
            s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    # pdb.set_trace()
    if not isinstance(queries, np.ndarray):
        queries = queries.to('cpu').numpy()
    if not isinstance(supports, np.ndarray):
        supports = supports.to('cpu').numpy()

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)

    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def input_package(pcd_list, prompt_ind_list, config, neighborhood_limits, device=None):
    """
    :param pcd_list: list，包含多个点云，每个点云是一个(N, D)的张量，N是点数，D是点维度(至少包含x, y, z坐标)。
    :param prompt_ind_list: list，包含与pcd_list中每个点云对应的提示点索引。
    :param config: Config对象，包含模型的配置参数，如卷积半径、层数等。
    :param neighborhood_limits: list，包含每一层的邻居数量限制。
    :param to_gpu: bool，如果为True，则将输入数据移动到GPU上。
    :return: dict_inputs: dict，包含模型输入所需的所有数据，
                          键包括'points'(点云数据xyz), 'neighbors'(邻居索引), 'pools'(池化索引),
                          'upsamples'(上采样索引), 'features'(点特征), 'stack_lengths'(每个批次的长度)。
    """

    batched_points_list = []  # (x,y,z)
    batched_features_list = []  # (r,g,b,label) label:1(except for one -1)
    batched_lengths_list = []  # point_number

    for i in range(len(pcd_list)):
        pcd = pcd_list[i]
        promt_ind = prompt_ind_list[i]  # (1)

        batched_points_list.append(pcd[:, :3])
        feats = torch.ones(pcd.shape[0], 1)
        feats = feats.to(device) if pcd.device.type != 'cpu' else feats
        feats[promt_ind] = -1
        feats = torch.cat([pcd[:, 3:6], feats], dim=1)
        batched_features_list.append(feats)

        batched_lengths_list.append(pcd.shape[0])

    batched_features = torch.cat(batched_features_list, dim=0)
    batched_points = torch.cat(batched_points_list, dim=0)
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []  # 当前层的所有块
    layer = 0  # 当前层索引

    # Lists of inputs
    input_points = []  # 输入的点
    input_neighbors = []  # 输入的邻居索引
    input_pools = []  # 输入的池化索引
    input_upsamples = []  # 输入的上采样索引
    input_batches_len = []  # 输入批次的长度

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        # 遇到全局池化或上采样时停止
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer,and continue when meeting layers except strided/pool
        # 收集当前层的所有块,当前层:strided/pool前
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            # neighbors
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius
            # first:0.1 dl: 0.2, 0.4
            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'points': input_points,  # 0: N0 (x,y,z)  1: N1 sub_(x,y,z)  2: N2 sub_(x,y,z)
        'neighbors': input_neighbors,  # 0: N0 neighbors_(x,y,z)  1: N1 neighbors_(x,y,z)  2: N2 neighbors_(x,y,z)
        'pools': input_pools,  # 0: N1 (0-1)neighbors  1: N2 (1-2)neighbors;  2:0 (0)
        'upsamples': input_upsamples,  # 0: N0 (1-0)neighbors  1: N1 (2-1)neighbors;  2:0 (0)
        'features': batched_features.float(),  # (x,y,z,label)
        'stack_lengths': input_batches_len  # [N0,N1,N2]
    }
    if device:
        for k, v in dict_inputs.items():
            if type(v) == list:
                dict_inputs[k] = [item.to(device) for item in v]
            else:
                dict_inputs[k] = v.to(device)

    return dict_inputs


class PSNet3d(nn.Module):

    def __init__(self):
        super(PSNet3d, self).__init__()

        config = Config()
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim  # 4
        out_dim = config.first_feats_dim  # 1024
        self.K = config.num_kernel_points

        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(
                block, r,
                in_dim, out_dim,
                layer, config
            ))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        out_dim = 256

        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consec utive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(
                block, r,
                in_dim, out_dim,
                layer, config)
            )

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        # self.feat_proj = nn.Linear(1024 + 3 + 256 + 3 + 256, 256, bias=True)
        self.feat_proj = nn.Linear(4614, 256, bias=True)

        self.pe_func_256 = PositionEmbeddingCoordsSine(3, 256)
        self.attns = nn.ModuleList()
        self.attn_layers = 6
        for i in range(self.attn_layers):
            self.attns.append(SelfAttention(dim=256, num_heads=8))

        self.config = config

    def forward(self, pcd_list, prompt_ind_list, device):
        # Tensor, cpu
        # x: dict {
        #     points: [n1x3, n2x3, n3x3]
        #     features: [n1x256, n2x512, n3x1024]
        #     stack_lengths: [(src_n1, tgt_n1), (src_n2, tgt_n2), (src_n3, tgt_n3)]
        # }
        inp = input_package(pcd_list, prompt_ind_list, self.config, self.config.neighborhood_limits, device)
        feats = inp['features'].clone().detach()#(xyzlabel)
        # for i in range(len(pcd_list)):
        #     print(inp['stack_lengths'][0][i].item(), inp['stack_lengths'][1][i].item(), inp['stack_lengths'][2][i].item())

        # encoder
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(feats)
            feats = block_op(feats, inp)

        feat_list = []
        pe_list = []
        prompt_point_list = []
        super_point_list = []
        prefix = 0
        pe = self.pe_func_256(inp['points'][-1]) #(xyz)
        for i in range(len(pcd_list)):
            length_i = inp['stack_lengths'][-1][i].item()
            feat_list.append(feats[prefix:prefix + length_i, :])
            pe_list.append(pe[prefix:prefix + length_i, :])
            prompt_point_list.append(
                pcd_list[i][prompt_ind_list[i], :3].view(1, 3).repeat([length_i, 1]).to(feats.device))
            super_point_list.append(inp['points'][-1][prefix:prefix + length_i, :])
            prefix += length_i

        prompt_point_list = torch.cat(prompt_point_list, dim=0)
        super_point_list = torch.cat(super_point_list, dim=0)
        feats = torch.cat([feats, prompt_point_list, self.pe_func_256(prompt_point_list), super_point_list,
                           self.pe_func_256(super_point_list)], dim=1)
        feats = self.feat_proj(feats)
        # print(feats.shape)

        feat_list = []
        prefix = 0
        for i in range(len(pcd_list)):
            length_i = inp['stack_lengths'][-1][i].item()
            feat_list.append(feats[prefix:prefix + length_i, :])
            prefix += length_i

        for i in range(self.attn_layers):
            for j in range(len(feat_list)):
                feat_list[j] = self.attns[i]((feat_list[j] + pe_list[j]).unsqueeze(0))[0]
        feats = torch.cat(feat_list, dim=0)

        # decoder
        # print(feats.shape)
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                feats = torch.cat([feats, skip_x.pop()], dim=1)
            feats = block_op(feats, inp)
        # print(feats.shape)
        feats = torch.sigmoid(feats)

        feat_list = []
        prefix = 0
        for i in range(len(pcd_list)):
            length_i = inp['stack_lengths'][0][i].item()
            feat_list.append(feats[prefix:prefix + length_i, :])
            prefix += length_i
        return feat_list

