import numpy as np
import torch
from mamba_ssm import TreeMamba, BatchedTree

class LeroNet(torch.nn.Module):
    def __init__(self, input_feature_dim: int) -> None:
        super().__init__()
        self.input_feature_dim = input_feature_dim
        dim = 64
        self.feature_proj = torch.nn.Linear(input_feature_dim, dim)
        self.model = TreeMamba(
            d_model=dim,
            d_state=64,
            expand=2,
        )
        self.cost_proj = torch.nn.Linear(dim, 1)

    def forward(self, batched_tree: BatchedTree) -> torch.Tensor:
        x = self.feature_proj(batched_tree.x)
        batched_tree.x = x
        y = self.model(batched_tree).squeeze()
        return self.cost_proj(y)

# xa = np.random.randn(6, 64).astype(np.float32)
# root_a = IndexTreeNode(0)
# root_a.append(1)
# root_a[-1].append(2)
# root_a[-1][-1].append(3)
# root_a[-1][-1].append(4)
# root_a.append(5)
# a = Tree(xa, root_a)

# xb = np.random.randn(7, 64).astype(np.float32)
# root_b = IndexTreeNode(0)
# root_b.append(1)
# root_b[-1].append(2)
# root_b[-1][-1].append(3)
# root_b[-1].append(4)
# root_b[-1][-1].append(5)
# root_b[-1][-1].append(6)
# b = Tree(xb, root_b)

# batched_trees = batch_trees([a, b])

# model = TreeMamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=64, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     expand=2,    # Block expansion factor
# ).to("cuda")

# y = model(batched_trees.to("cuda"))
# print(y.shape)

# torch.save(model.state_dict(), "model.pth")