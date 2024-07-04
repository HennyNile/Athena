import sys

import torch
from torch import nn

sys.path.append('.')
from src.utils.TreeConvolution.tcnn import (BinaryTreeConv, MaxPoolingAll,
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
        self.pooling = MaxPoolingAll()
        self.card_estimator1 = nn.Linear(64, 32)
        self.card_estimator2 = nn.Linear(32, 1)
        self.cost_estimator1 = nn.Linear(96, 32)
        self.cost_estimator2 = nn.Linear(32, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, trees, nodes):
        x, _ = self.tree_conv(trees)
        x = self.pooling(x, nodes)
        cards_info = self.activation(self.card_estimator1(x))
        cards = self.card_estimator2(cards_info)
        plan_with_cards = torch.cat((x, cards_info), dim=2)
        cost = self.activation(self.cost_estimator1(plan_with_cards))
        cost = self.cost_estimator2(cost)[:, 0]
        return cost.squeeze(), cards.squeeze()