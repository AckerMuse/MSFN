import torch
from torch import nn
import torch.nn.functional as F
from .torch_nn import MLP, act_layer, norm_layer, BondEncoder
from .torch_edge import DilatedKnnGraph


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)  # elu-激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GATConv(nn.Module):
    """
    Graph Attention Convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels,  act='relu', norm=None,dropout=0.6, alpha=0.2, bias=True, heads=8):
        super(GATConv, self).__init__()
        self.gconv = GraphAttentionLayer(in_channels, out_channels,dropout, alpha, concat=True)
        m =[]
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, adj):
        out = self.unlinear(self.gconv(x, adj))
        return out


import torch.nn.init as init
import math

class GraphConvolution(nn.Module):#GCN层
    """
    GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features#输入特征
        self.out_features = out_features#输出特征
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # 根据input_features, out_features构造weight参数
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))#构造bias参数
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#对bias进行随机初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # GraphConvolution forward：input*weight
        output = torch.spmm(adj, support)  # 稀疏矩阵的相乘，spmm和mm都是矩阵相乘，adj*support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SemiGCNConv(nn.Module):
    """
    SemiGCN convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = GraphConvolution(in_channels, out_channels)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, adj):
        out = self.unlinear(self.gconv(x, adj))
        return out


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv,
                 act='relu', norm=None, bias=True, heads=8):
        super(GraphConv, self).__init__()
        if conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)

        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, adj):
        return self.gconv(x, adj)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class PlainDynBlock(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(PlainDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index), batch


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale


    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index) + x*self.res_scale, batch


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(in_channels, out_channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        dense = self.body(x, batch, edge_index)
        return torch.cat((x, dense), 1), batch


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def  __init__(self, channels1,channels2,conv='gcn', act='relu', norm=None, bias=True, heads=8,  res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels1, channels2, conv, act, norm, bias, heads)
        # self.fusion_block = MLP([channels2, channels1], 'leakyrelu', None, bias=True)
        self.res_scale = res_scale

    def forward(self, x, adj):
        # return self.fusion_block(self.body(x,adj)) + x*self.res_scale
        return self.body(x, adj) + x * self.res_scale

class DenseGraphBlock(nn.Module):
    """
    Dense Static graph convolution block
    """
    def __init__(self, channels1, channels2, conv='gcn', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(channels1, channels2, conv, act, norm, bias, heads)

    def forward(self, x, adj):
        dense = self.body(x, adj)
        return torch.cat((x, dense), 1)

