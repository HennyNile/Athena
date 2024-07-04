import torch
import torch.nn as nn

class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data):
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        zero_vec = zero_vec.to(results.device)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)

class ChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.weights = nn.Linear(in_channels, out_channels)

    def forward(self, flat_data):
        trees, _ = flat_data
        trees = self.weights(trees[:,:,1:].transpose(1, 2)).transpose(1, 2)
        batch_size, dim, _ = trees.shape
        return torch.cat((torch.zeros(batch_size, dim, 1, dtype=torch.float32, device=trees.device), trees), dim=2)

class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]), x[1])

class TreeLayerNorm(nn.Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)

class DynamicPooling(nn.Module):
    def forward(self, x):
        return torch.max(x[0], dim=2).values

class PreciseMaxPooling(nn.Module):
    def forward(self, x, nodes):
        # x    : (batch_size, dim,           max_num_nodes + 1)
        # nodes: (batch_size, max_num_nodes, max_num_nodes + 1)
        _, dim, _ = x.shape
        top_nodes = nodes[:, 0, :]
        top_nodes = top_nodes.unsqueeze(1).repeat(1, dim, 1)
        x = x.masked_fill(~top_nodes, -torch.inf)
        return torch.max(x, dim=2).values

class MaxPoolingAll(nn.Module):
    def forward(self, x, nodes):
        vecs = []
        _, dim, _ = x.shape
        _, num_max_nodes, _ = nodes.shape
        for i in range(num_max_nodes):
            nodes_i = nodes[:, i, :]
            nodes_i = nodes_i.unsqueeze(1).repeat(1, dim, 1)
            masked_x = x.masked_fill(~nodes_i, -torch.inf)
            vecs.append(torch.max(masked_x, dim=2).values)
        return torch.stack(vecs, dim=1)