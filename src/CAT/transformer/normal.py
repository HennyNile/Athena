import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import *

class Attention(nn.Module):
    def __init__(self, args: AttentionArgs, max_seq: int = 128) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.use_rope = args.use_rope

        if args.use_rope:
            theta = 10000.0
            theta_of_dims = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            pos = torch.arange(max_seq).float().unsqueeze(1)
            freqs = pos * theta_of_dims
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(1)
            self.register_buffer('freqs_cis', freqs_cis, persistent=False)

        self.wq = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wk = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wv = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wo = nn.Linear(self.dim, self.dim, bias=args.bias)

    def forward(self, x: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        # x   : (batch_size, seq_len, dim)
        # pos : torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        # mask: (batch_size, seq_len, seq_len)

        batch_size, seq_len, _ = x.shape
        mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=x.device)
        for idx, l in enumerate(seq_lens):
            mask[idx, :l, l:] = -torch.inf
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (batch_size, seq_len, dim)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

        if self.use_rope:
            # freqs = pos * self.theta_of_dims
            # (seq_len, head_dim // 2)
            # freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(1)
            # (seq_len, 1, head_dim // 2)
            xq = xq.reshape(*xq.shape[:-1], -1, 2)
            xk = xk.reshape(*xk.shape[:-1], -1, 2)
            # (batch_size, seq_len, n_heads, head_dim // 2, 2)
            xq = torch.view_as_complex(xq)
            xk = torch.view_as_complex(xk)
            # (batch_size, seq_len, n_heads, head_dim // 2)
            xq = torch.view_as_real(xq * self.freqs_cis[:seq_len]).flatten(3)
            xk = torch.view_as_real(xk * self.freqs_cis[:seq_len]).flatten(3)
            # (batch_size, seq_len, n_heads, head_dim)


        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # (batch_size, n_heads, seq_len, head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print(f"scores: {scores[:,0,:,:]}")
        scores = scores + mask.unsqueeze(1)
        # print(f"scores + mask: {scores[:,0,:,:]}")
        scores = F.softmax(scores, dim=-1)
        # print(f"softmax(scores): {scores[:,0,:,:]}")
        # scores = scores.nan_to_num()
        # (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(scores, xv)
        # (batch_size, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (batch_size, seq_len, dim)

        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, attn_args: AttentionArgs, ffn_args: FeedForwardArgs, max_seq: int = 128) -> None:
        super().__init__()
        self.attn = Attention(attn_args, max_seq)
        self.ffn = FeedForward(ffn_args)
        self.ln1 = nn.LayerNorm(attn_args.dim)
        self.ln2 = nn.LayerNorm(attn_args.dim)

    def forward(self, x: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        h = x + self.attn(self.ln1(x), seq_lens)
        return h + self.ffn(self.ln2(h))

class Encoder(nn.Module):
    def __init__(self, args: TransformerArgs, max_seq: int = 128) -> None:
        super().__init__()
        self.output_vec = nn.Parameter(torch.randn((args.embedding_dim,), dtype=torch.float))
        self.embed = nn.Linear(args.feature_dim, args.embedding_dim, args.bias)
        self.blocks = nn.ModuleList([TransformerBlock(args.get_attn_args(), args.get_ffn_args(), max_seq) for _ in range(args.n_layers)])

    def forward(self, x: torch.Tensor, seq_lens: list[int]):
        x = self.embed(x)
        # x = torch.concat((self.output_vec.repeat(x.shape[0], 1, 1), x), dim=1)
        # seq_lens = [l + 1 for l in seq_lens]
        batch_size, seq_len, _ = x.shape
        mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32, device=x.device)
        for idx, l in enumerate(seq_lens):
            mask[idx, :l] = 1.
        for block in self.blocks:
            x = block(x, seq_lens)
        return torch.sum(x * mask, dim=1) / torch.sum(mask, dim=1)