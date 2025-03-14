#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020

import time
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

from torch.utils.data import Dataset, DataLoader
from .kernel_points import load_kernels
import numpy as np


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    实现自定义的 gather 操作以加快反向传播。
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    所有方法功能一致，都是为idx中每个值寻找了x[idx][:]的张量
    即： (NN)[(NN)]->NNN，[i,j]=x[idx[i,j]]
    测试之下三种方法速度最快、支持数据量最大的都为方法0，但方法2貌似更为灵活
    """
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))  # 扩展到‘1’维度
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))  # 扩展到‘2’维度
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    计算距离的高斯函数（距离的高斯）
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    使用高斯公式计算距离的高斯值，其中sq_r为半径，sig为标准差，eps为防止除零的小数
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    从最近邻处汇集特征。警告：这个函数假设邻居是有序的。
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    # 添加一行最小特征值（全0的值），以防有些池化位置没有对应的邻居
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    # 使用gather函数，根据inds的第一列索引来获取每个池化位置的特征，形状为 [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    # 添加一行最小特征值（全0的值），以防有些池化位置没有对应的邻居
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    # 获取每个池化位置的所有特征 [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    # 返回最大值[n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        # 对每个批次云的特征进行平均
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    # 将每批次的平均特征进行堆叠
    return torch.stack(averaged_features)


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.核点的数量。
        :param p_dim: dimension of the point space.点空间的维度。
        :param in_channels: dimension of input features.输入特征的维度。
        :param out_channels: dimension of output features.输出特征的维度。
        :param KP_extent: influence radius of each kernel point.每个核点的影响半径。
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
                       用于初始化核心点的半径。对于可变形的KPConv，使用config.conv_radius。
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
                                    固定某些核心点的位置（'none', 'center'或'verticals'）。
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
                             核心点的影响函数（'constant', 'linear', 'gaussian'）。
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
                                 选择影响的聚合模式，是求和('sum')还是仅保留最近的('closest')。
        :param deformable: choose deformable or not 选择是否可变形。
        :param modulated: choose if kernel weights are modulated in addition to deformed 选择可变形的同时还调制核心权重。
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        # 运行变量，包含形变的KP与输入点的距离（用于正则化损失）
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        # 初始化权重
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Initiate weights for offsets
        # 如果可变形，则初始化偏移量的权重
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K

            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,  # out_channels
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)

            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        # 重置参数
        self.reset_parameters()

        # Initialize kernel points
        # 初始化核心点
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent

        K_points_numpy = load_kernels(self.radius,  # 1
                                      self.K,  # 15
                                      dimension=self.p_dim,  # 3
                                      fixed=self.fixed_kernel_points)
        # Parameter将该参数作为模型参数保留，可以被保存和加载
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # 前向传播函数：计算点云的核点卷积
        # q_pts: 查询点的坐标 [n_points, dim]
        # s_pts: 邻域点的坐标 [n_points, dim]
        # neighb_inds: 邻居索引 [n_points, n_neighbors]
        # x: 点特征 [n_points, in_channels]

        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            # 使用KPConv计算偏移特征，然后加上偏移偏置
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                # 如果启用调制，从特征中分离出未缩放的偏移量和调制因子
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                # 未启用调制，则所有特征均为偏移量
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            # 将偏移量缩放到本层的范围
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        # 为邻域添加一个假点
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        # 获取邻域点的坐标[n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        # 中心化邻域点
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        # 如果是可变形卷积，应用偏移量到核点上
        if self.deformable:
            self.deformed_KP = self.kernel_points + offsets
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        # 计算邻居点与变形核点之间的差异 [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        # 计算平方距离和 [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        # 对于可变形KP，通过忽略超出变形KP范围的点进行优化
        if self.deformable:

            # Save distances for loss
            # 保存最小距离平方，用于后续损失计算
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            # 计算每个点的邻点是否在核点范围内 [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)

            # New value of max neighbors
            # 获取新的最大的邻点
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            # 对每一行的邻点，获取范围内的索引 [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            # 收集新的邻点索引 [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            # 收集新的到KP的距离 [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            # 新的阴影邻点需要指向最后的阴影点
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        # 获取核点影响 [n_points, n_kpoints, n_neighbors]，根据KP影响方式（常数、线性、高斯）计算权重
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            # 'constant'常数的每个点的影响为1
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            # 'linear'近邻中只有最近的KP能影响每个点
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            # 'gaussian'高斯距离为核点
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        # 如果为最近模式，只有最近的KP能影响每个点
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        # 为阴影邻点添加一个零特征
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        # 获取每个邻点的特征 [n_points, n_neighbors, in_channels]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        # 应用距离权重 [n_points, n_kpoints, in_channels]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        # 应用调制
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        # 应用网络权重 [n_kpoints, n_points, out_channels]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        # 卷积求和 [n_points, out_channels]
        # return torch.sum(kernel_outputs, dim=0)
        output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

        # normalization term.
        # 正规化项
        neighbor_features_sum = torch.sum(neighb_x, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_features = output_features / neighbor_num.unsqueeze(1)

        return output_features

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, extent: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                                              self.KP_extent,
                                                                                              self.in_channels,
                                                                                              self.out_channels)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    """
    根据给定的参数和配置决定使用哪种类型的网络块。
    :param block_name: 网络块的名称，指定所需的块类型。
    :param radius: 用于某些类型的网络块，例如KPConv中的半径参数。
    :param in_dim: 输入特征的维度。
    :param out_dim: 输出特征的维度。
    :param layer_ind: 层的索引，用于某些特定的网络块。
    :param config: 包含其他所有必要配置的配置对象。
    :return: 创建的网络块实例。
    """
    if block_name == 'unary':
        # mlp+bn+relu
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    if block_name == 'last_unary':
        # mlp
        return LastUnaryBlock(in_dim, config.final_feats_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        初始化一个批量归一化块。如果网络不使用批量归一化，则用偏置替换。(注：张量为二维的)
        （N,C,L）-->nn.BatchNorm1d(C)
        (N,C)-->BatchNormBlock(C,True,0.1)
        use_bn=True,使用归一化；use_bn=False，则data + self.bias
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            # self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:
            x = x.unsqueeze(2)  # 在第三维度上增加一个维度
            x = x.transpose(0, 2)  # 交换第0维和第2维
            x = self.batch_norm(x)  # 应用批量归一化
            x = x.transpose(0, 2)  # 再次交换第0维和第2维，还原维度顺序
            return x.squeeze()  # 移除单维度条目
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        初始化一个标准的一元块，可以选择包括ReLU和BatchNorm。输入特征的维度。(二维张量)
        UnaryBlock(in,out,use_bn,0.1)-->mlp+bn+relu
        :param in_dim: dimension input features 输入特征的维度。
        :param out_dim: dimension input features 输出特征的维度。
        :param use_bn: boolean indicating if we use Batch Norm 布尔值，指示是否使用批量归一化。
        :param bn_momentum: Batch norm momentum 批量归一化的动量。
        :param no_relu: 布尔值，如果为True，则不使用ReLU激活函数，默认为False。
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        # linear处理的最后一维（*，*，in）-->（*，*，out）
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):  # 定义函数的打印方式
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class LastUnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard last_unary block without BN, ReLU.
        初始化一个标准的最后块，仅包括mlp(二维张量)
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(LastUnaryBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        return x

    def __repr__(self):
        return 'LastUnaryBlock(in_feat: {:d}, out_feat: {:d})'.format(self.in_dim,
                                                                      self.out_dim)


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        初始化一个简单的卷积块带有relu和bn
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        # 根据当前半径计算KP_extent，KP_extent是核点的影响半径
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        # 获取其他配置参数
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        # 定义KPConv层，可能包括可变形版本
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other opperations
        # 定义BatchNorm层和LeakyReLU激活函数
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):
        """
         前向传播方法。
         :param x: 输入特征张量。
         :param batch: 包含点云数据和邻接信息的字典。
         :return: 经过卷积块处理后的特征张量。
         """
        # 根据块名称选择查询点和支持点
        if 'strided' in self.block_name:
            # 对于步长版本，查询点和支持点来自不同的层
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            # 对于非步长版本，查询点和支持点来自同一层
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        # 应用KPConv，然后返回BatchNorm和LeakyReLU
        return self.leaky_relu(self.batch_norm(x))


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        初始化一个残差瓶颈卷积块。
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        # 首先通过一个用于降维的一元块
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()  # 占位符

        # KPConv block
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        # 其次通过一个用于升维的一元块
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        # 如果输入和输出维度不同，添加一个快捷连接的一元块
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()  # 占位符

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        # First downscaling mlp
        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        初始化一个全局平均块
        """
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch['stack_lengths'][-1])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        初始化一个最近邻上采样块
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch['upsamples'][self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        初始化一个最大池化块
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch['pools'][self.layer_ind + 1])


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.num_layers = 3
            self.in_points_dim = 3
            self.first_feats_dim = 256
            self.first_subsampling_dl = 0.04
            self.in_feats_dim = 4
            self.final_feats_dim = 1
            self.conv_radius = 2.75
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
            self.neighborhood_limits = [96, 48, 48]

            self.architecture = [
                'simple',
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
                'unary',
                'nearest_upsample',
                'unary',
                'last_unary'
            ]


    data = [np.column_stack((np.random.uniform(0, 10, 10),
                             np.random.uniform(0, 10, 10),
                             np.random.uniform(0, 10, 10)))
            for _ in range(10)]


    class XYZ(Dataset):
        def __init__(self, data):
            self.data = [torch.from_numpy(xyz) for xyz in data]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            config = Config()
            block = config.architecture[0]
            self.block1 = block_decider(block, config.first_subsampling_dl * config.conv_radius, 10, 5, 0, config)

        def forward(self, x, batch):
            x = self.block1(x, batch)
            return x


    dataset = XYZ(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    model = Net()

    for batch, data in enumerate(dataloader):
        pre = model(data, batch)
        print(pre)

    print('p')
