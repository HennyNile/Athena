import torch

from torch import nn
import torch.nn.functional as F

from transformer.common import TransformerArgs
from transformer.normal import Encoder
from transformer.tree import TreeEncoder

class CatArgs:
    def __init__(
            self,
            feature_dim: int,
            expr_embedding_dim: int,
            expr_hidden_dim: int,
            expr_n_heads: int,
            expr_n_layers: int,
            node_feature_dim: int,
            node_embedding_dim: int,
            node_hidden_dim: int,
            node_n_heads: int,
            node_n_layers: int,
            plan_embedding_dim: int,
            plan_hidden_dim: int,
            plan_n_heads: int,
            plan_n_layers: int,
            card_info_dim: int,
            cost_embedding_dim: int,
            cost_hidden_dim: int,
            cost_n_heads: int,
            cost_n_layers: int,
            bias: bool = True):
        self.feature_dim = feature_dim
        self.expr_embedding_dim = expr_embedding_dim
        self.expr_hidden_dim = expr_hidden_dim
        self.expr_n_heads = expr_n_heads
        self.expr_n_layers = expr_n_layers
        self.node_feature_dim = node_feature_dim
        self.node_embedding_dim = node_embedding_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_n_heads = node_n_heads
        self.node_n_layers = node_n_layers
        self.plan_embedding_dim = plan_embedding_dim
        self.plan_hidden_dim = plan_hidden_dim
        self.plan_n_heads = plan_n_heads
        self.plan_n_layers = plan_n_layers
        self.card_info_dim = card_info_dim
        self.cost_embedding_dim = cost_embedding_dim
        self.cost_hidden_dim = cost_hidden_dim
        self.cost_n_heads = cost_n_heads
        self.cost_n_layers = cost_n_layers
        self.bias = bias

class CatModel(nn.Module):
    def __init__(self, args: CatArgs):
        super().__init__()
        cost_feature_dim = args.node_embedding_dim + args.card_info_dim
        expr_args = TransformerArgs(args.feature_dim, args.expr_embedding_dim, args.expr_hidden_dim, args.expr_n_heads, args.expr_n_layers, args.bias)
        node_args = TransformerArgs(args.node_feature_dim, args.node_embedding_dim, args.node_hidden_dim, args.node_n_heads, args.node_n_layers, args.bias, use_rope=False)
        plan_args = TransformerArgs(args.node_embedding_dim, args.plan_embedding_dim, args.plan_hidden_dim, args.plan_n_heads, args.plan_n_layers, args.bias)
        cost_args = TransformerArgs(cost_feature_dim, args.cost_embedding_dim, args.cost_hidden_dim, args.cost_n_heads, args.cost_n_layers, args.bias)
        self.expr_encoder = Encoder(expr_args)
        self.node_encoder = Encoder(node_args)
        self.plan_encoder = TreeEncoder(plan_args)
        self.card_predictor = nn.Linear(args.plan_embedding_dim, args.card_info_dim, bias=args.bias)
        self.activation = nn.LeakyReLU()
        self.cost_predictor = TreeEncoder(cost_args)
        self.batch_norm = nn.BatchNorm1d(args.cost_embedding_dim)
        self.card_estimator = nn.Linear(args.card_info_dim, 1, bias=args.bias)
        self.cost_estimator = nn.Linear(args.cost_embedding_dim, 1, bias=args.bias)

    def forward(self, x, pos, mask):
        plan = self.plan_encoder(x, pos, mask)
        card_info = self.card_predictor(plan)
        cards = self.card_estimator(self.activation(card_info)).flatten(1)
        plan_with_card = torch.cat([plan, card_info], dim=2)
        cost_info = self.cost_predictor(plan_with_card, pos, mask)
        cost = self.cost_estimator(self.batch_norm(cost_info[:, 0]))
        return cost, cards
    
    def cards_output(self, x, pos, mask):
        plan = self.plan_encoder(x, pos, mask)
        card_info = self.card_predictor(plan)
        cards = self.card_estimator(self.activation(card_info)).flatten(1)
        return cards

    def train_output(self, x, pos, mask, pretrain: bool = False):
        plan = self.plan_encoder(x, pos, mask)
        card_info = self.card_predictor(plan)
        cards = self.card_estimator(self.activation(card_info)).flatten(1)
        plan_with_card = torch.cat([plan.detach(), card_info.detach()], dim=2)
        cost_info = self.cost_predictor(plan_with_card, pos, mask)
        if not pretrain:
            cost = self.cost_estimator(self.batch_norm(cost_info[:, 0]))
        else:
            cost = self.cost_estimator(cost_info).flatten(1)
        return cost, cards
