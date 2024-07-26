import numpy as np
import torch
from torch import nn
from mamba_ssm import TreeMamba2, BatchedTree2

class GatedFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x: BatchedTree2) -> BatchedTree2:
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, has_ff=True, d_state=16, expand=2):
        super().__init__()
        self.has_ff = has_ff
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = TreeMamba2(d_model=d_model, d_state=d_state, expand=expand)
        if self.has_ff:
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = GatedFeedForward(d_model, expand * d_model)

    def forward(self, batched_trees: BatchedTree2):
        x = batched_trees.x
        batched_trees.x = self.norm1(x)
        y = self.mamba(batched_trees)
        x = y + x
        if self.has_ff:
            y = self.norm2(x)
            y = self.ff(y)
            x = y + x
        batched_trees.x = x
        return batched_trees

class LeroNet(torch.nn.Module):
    def __init__(self, input_feature_dim: int) -> None:
        super().__init__()
        self.input_feature_dim = input_feature_dim
        num_layer = 2
        self.dim = 64
        self.feature_proj = torch.nn.Linear(input_feature_dim, self.dim)
        self.mamba_layers = nn.ModuleList([MambaBlock(self.dim, has_ff=i != num_layer - 1) for i in range(num_layer)])
        self.batch_norm = nn.BatchNorm1d(2 * self.dim)
        self.cost_proj = torch.nn.Linear(2 * self.dim, self.dim)
        self.output_act = nn.LeakyReLU()
        self.cost_est = nn.Linear(self.dim, 1)
        self.cost_skip = nn.Linear(2 * self.dim, 1)

    def forward(self, batched_trees: BatchedTree2) -> torch.Tensor:
        batched_trees.x = self.feature_proj(batched_trees.x)
        for m in self.mamba_layers:
            batched_trees = m(batched_trees)
        y = batched_trees.x
        y = y[torch.arange(y.shape[0], device=y.device),batched_trees.output_indices,:]
        y = y.view(-1, 2 * self.dim)
        y = self.batch_norm(y)
        y = self.output_act(self.cost_proj(y))
        return self.cost_est(y)