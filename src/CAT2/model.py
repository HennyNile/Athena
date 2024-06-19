import sys

import torch
import torch.nn as nn

sys.path.append('.')
from src.utils.transformer.common import TransformerArgs
from src.utils.transformer.tree import TreeEncoder

class CatModel(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int, nhead: int, hidden_dim: int, num_layers: int, dropout: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.embed = nn.Linear(feature_dim, embed_dim)
        transformer_arg1 = TransformerArgs(
            embed_dim,
            hidden_dim,
            nhead,
            num_layers
        )
        self.encoder = TreeEncoder(transformer_arg1)
        self.card_estimator1 = nn.Linear(embed_dim, embed_dim // 2)
        self.card_estimator2 = nn.Linear(embed_dim // 2, 1)
        self.activation = nn.LeakyReLU()
        transformer_arg2 = TransformerArgs(
            embed_dim * 3 // 2,
            hidden_dim * 3 // 2,
            nhead * 3 // 2,
            1
        )
        self.cost_encoder = TreeEncoder(transformer_arg2)
        self.batch_norm = nn.BatchNorm1d(embed_dim * 3 // 2)
        self.cost_estimator = nn.Linear(embed_dim * 3 // 2, 1)

    def node_repr(self, x: torch.Tensor, pos: torch.Tensor):
        line_idx = torch.arange(x.shape[0], device=x.device).view(-1, 1)
        return x[line_idx, pos]

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, output_idx: torch.Tensor):
        _, seq_len, _ = x.shape
        x = self.embed(x)
        x = self.encoder(x, pos, mask)
        x = self.node_repr(x, output_idx)
        return x

    def cards_output(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, output_idx: torch.Tensor):
        plan = self(x, pos, mask, output_idx)
        cards_info = self.card_estimator1(plan)
        cards = self.card_estimator2(self.activation(cards_info)).flatten(1)
        return cards

    def cost_output(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, node_pos: torch.Tensor, node_mask: torch.Tensor, output_idx: torch.Tensor):
        plan = self(x, pos, mask, output_idx)
        cards_info = self.card_estimator1(plan)
        plan_with_cards = torch.cat([plan.detach(), cards_info.detach()], dim=2)
        cost_info = self.cost_encoder(plan_with_cards, node_pos, node_mask)
        cost = self.cost_estimator(self.batch_norm(cost_info[:, 0]))
        return cost

    def cost_and_cards_output(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, node_pos: torch.Tensor, node_mask: torch.Tensor, output_idx: torch.Tensor):
        plan = self(x, pos, mask, output_idx)
        cards_info = self.card_estimator1(plan)
        plan_with_cards = torch.cat([plan, cards_info.detach()], dim=2)
        cost_info = self.cost_encoder(plan_with_cards, node_pos, node_mask)
        cost = self.cost_estimator(self.batch_norm(cost_info[:, 0]))
        cards = self.card_estimator2(self.activation(cards_info)).flatten(1)
        return cost, cards