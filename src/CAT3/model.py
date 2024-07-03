import sys

from torch import nn

sys.path.append('.')
from src.utils.TreeConvolution.tcnn import (BinaryTreeConv, PreciseMaxPooling,
                                  TreeActivation, TreeLayerNorm, ChannelMixer)

class LeroBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.conv1 = BinaryTreeConv(input_dim, output_dim)
        self.conv2 = ChannelMixer(input_dim, output_dim)
        self.norm = TreeLayerNorm()
        self.activation = TreeActivation(nn.LeakyReLU())

    def forward(self, flat_data):
        y = self.conv1(flat_data)
        y = self.norm(y)
        y_tree, y_indices = y
        x_tree = self.conv2(flat_data)
        return self.activation((x_tree + y_tree, y_indices))

class LeroNet(nn.Module):
    def __init__(self, input_feature_dim, expr_dim) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.expr_dim = expr_dim
        self.expr_encoder = nn.Linear(expr_dim, 64)
        self.tree_conv = nn.Sequential(
            LeroBlock(self.input_feature_dim + 2 * 64, 256),
            LeroBlock(256, 128),
            LeroBlock(128, 64)
        )
        self.pooling = PreciseMaxPooling()
        self.estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, trees, nodes):
        x, _ = self.tree_conv(trees)
        x = self.pooling(x, nodes)
        return self.estimator(x)