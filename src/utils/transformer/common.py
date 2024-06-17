import torch
from torch import nn
import torch.nn.functional as F

class AttentionArgs:
    def __init__(self, dim: int, n_heads: int, bias: bool = False, use_rope: bool = True) -> None:
        self.dim = dim
        self.n_heads = n_heads
        self.bias = bias
        self.use_rope = use_rope
        assert self.dim % self.n_heads == 0, 'dim must be divisible by n_heads'
        assert (self.dim // self.n_heads) % 8 == 0, 'head_dim must be divisible by 8'

class FeedForwardArgs:
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.bias = bias

class TransformerArgs:
    def __init__(self, embedding_dim: int, hidden_dim: int, n_heads: int, n_layers: int, bias: bool = False, use_rope: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.bias = bias
        self.use_rope = use_rope

    def get_attn_args(self) -> AttentionArgs:
        return AttentionArgs(self.embedding_dim, self.n_heads, self.bias, self.use_rope)
    
    def get_ffn_args(self) -> FeedForwardArgs:
        return FeedForwardArgs(self.embedding_dim, self.hidden_dim, self.bias)

class FeedForward(nn.Module):
    def __init__(self, args: FeedForwardArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=args.bias)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=args.bias)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))