import torch

from torch import nn
import torch.nn.functional as F

from src.utils.transformer.common import TransformerArgs
from src.utils.transformer.normal import Encoder
from src.utils.transformer.tree import TreeEncoder

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

class EmbedEncoder(nn.Module):
    def __init__(self, feature_dim: int, args: TransformerArgs):
        super().__init__()
        self.embed = nn.Linear(feature_dim, args.embedding_dim, bias=args.bias)
        self.encoder = Encoder(args)
    
    def forward(self, x: torch.Tensor, seq_lens: list[int]):
        x = self.embed(x)
        x = self.encoder(x, seq_lens)
        return x

class CatModel(nn.Module):
    def __init__(self, args: CatArgs):
        super().__init__()
        cost_feature_dim = args.node_embedding_dim + args.card_info_dim
        expr_args = TransformerArgs(args.expr_embedding_dim, args.expr_hidden_dim, args.expr_n_heads, args.expr_n_layers, args.bias)
        node_args = TransformerArgs(args.node_embedding_dim, args.node_hidden_dim, args.node_n_heads, args.node_n_layers, args.bias, use_rope=False)
        plan_args = TransformerArgs(args.plan_embedding_dim, args.plan_hidden_dim, args.plan_n_heads, args.plan_n_layers, args.bias)
        cost_args = TransformerArgs(args.cost_embedding_dim, args.cost_hidden_dim, args.cost_n_heads, args.cost_n_layers, args.bias)
        self.expr_encoder = EmbedEncoder(args.feature_dim, expr_args)
        self.node_encoder = EmbedEncoder(args.node_feature_dim, node_args)
        self.plan_embed = nn.Linear(args.node_embedding_dim, args.plan_embedding_dim, bias=args.bias)
        self.plan_encoder = TreeEncoder(plan_args)
        self.card_predictor = nn.Linear(args.plan_embedding_dim, args.card_info_dim, bias=args.bias)
        self.activation = nn.LeakyReLU()
        self.cost_embed = nn.Linear(cost_feature_dim, args.cost_embedding_dim, bias=args.bias)
        self.cost_predictor = TreeEncoder(cost_args)
        self.batch_norm = nn.BatchNorm1d(args.cost_embedding_dim)
        self.card_estimator = nn.Linear(args.card_info_dim, 1, bias=args.bias)
        self.cost_estimator = nn.Linear(args.cost_embedding_dim, 1, bias=args.bias)

    def cards_output(self, x, pos, mask):
        plan = self.plan_encoder(self.plan_embed(x), pos, mask)
        card_info = self.card_predictor(plan)
        cards = self.card_estimator(self.activation(card_info)).flatten(1)
        return cards

    def cost_output(self, x, pos, mask):
        plan = self.plan_encoder(self.plan_embed(x), pos, mask)
        card_info = self.card_predictor(plan)
        plan_with_card = torch.cat([plan.detach(), card_info.detach()], dim=2)
        cost_info = self.cost_predictor(self.cost_embed(plan_with_card), pos, mask)
        cost = self.cost_estimator(self.batch_norm(cost_info[:, 0]))
        return cost