import numpy as np
import torch
from torch import nn
from mamba_ssm import TreeMamba, BatchedTree

class LeroNet(torch.nn.Module):
    def __init__(self, input_feature_dim: int) -> None:
        super().__init__()
        self.input_feature_dim = input_feature_dim
        input_rank = 12
        num_layer = 1
        dim = 64
        state = 64
        self.low_rank_proj = torch.nn.Linear(input_feature_dim, input_rank)
        self.feature_proj = torch.nn.Linear(input_rank, dim)
        self.mamba_layers = nn.ModuleList([TreeMamba(d_model=dim, d_state=state, expand=2) for _ in range(num_layer)])
        self.cost_proj = torch.nn.Linear(dim, dim)
        self.output_act = nn.LeakyReLU()
        self.cost_est = nn.Linear(dim, 1)

    def forward(self, batched_tree: BatchedTree) -> torch.Tensor:
        x = self.feature_proj(self.low_rank_proj(batched_tree.x))
        batched_tree.x = x
        for m in self.mamba_layers:
            y = m(batched_tree)
            batched_tree.x = y
        y = self.output_act(self.cost_proj(y[:,0,:]))
        return self.cost_est(y)