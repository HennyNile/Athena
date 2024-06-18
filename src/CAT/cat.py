import argparse
import math
import os
import re
import itertools

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np

from libcat.tokenizer import InputTokenType, SyntaxType, tokenizer, featurize_exprs
from libcat.node_encoder import KeyType, EmbeddingManager, featurize_nodes, batch_seqs

from cat_model import CatModel, CatArgs

def get_alias_map(sample):
    count_map = {}
    alias_map = {}
    relation_names = set()
    index_names = set()
    root = sample['Plan']
    max_card = 0

    def dfs(node):
        if 'Index Name' in node:
            index = node['Index Name']
            index_names.add(index)
        if 'Relation Name' in node:
            rel = node['Relation Name']
            relation_names.add(rel)
            if 'Alias' in node:
                alias = node['Alias']
                count = count_map.get(rel, 0)
                alias_map[alias] = (rel, count)
                count_map[rel] = count + 1
        if 'Actual Rows' in node and 'Actual Loops' in node:
            card = node['Actual Rows'] * node['Actual Loops']
            nonlocal max_card
            max_card = max(max_card, card)
        for child in node.get('Plans', []):
            dfs(child)

    dfs(root)
    return alias_map, relation_names, max_card

class Sample:
    def __init__(self, plan: dict, alias_map: dict[str, str], exprs: list[torch.Tensor]) -> None:
        self.plan = plan
        self.alias_map = alias_map
        self.exprs = exprs

class Cat:
    def __init__(self, db_info):
        self.table_map = db_info.table_map
        self.column_map = db_info.column_map
        self.normalizer = db_info.normalizer
        self.word_table: dict[str, int] = {}
        self.max_table_idx: int = 0
        self.max_card: float = 0.
        self.feature_dim: int = 0
        self.num_input_types: int = len(InputTokenType)
        self.expr_embedding_dim = 256
        self.model: torch.nn.Module = None

    def fit_all(self, dataset: list[list[dict]]) -> None:
        # word_table and max_table_idx and feature_dim
        plans = [sample for query in dataset for sample in query]
        for sample in plans:
            alias_map, rel_names, _ = get_alias_map(sample)
            self._fit_sample(sample, alias_map, rel_names)
        num_input_types = len(InputTokenType)
        num_syntax_types = len(SyntaxType)
        vocab_size = len(self.word_table)
        num_tables = len(self.table_map)
        num_table_indices = self.max_table_idx + 1
        num_columns = len(self.column_map)
        number_dim = 1
        self.feature_dim = num_input_types + max(num_syntax_types, num_tables, num_table_indices, num_columns, vocab_size, number_dim)
        self.num_input_types = num_input_types
        print(f'feature_dim: {self.feature_dim}')

    def fit_train(self, dataset: list[list[dict]]) -> None:
        # max_card
        plans = [sample for query in dataset for sample in query]
        for sample in plans:
            _, _, max_card = get_alias_map(sample)
            self.max_card = max(self.max_card, float(max_card))
        print(f'max_card: {self.max_card}')

    def init_model(self) -> None:
        model_arg = CatArgs(
            feature_dim=self.feature_dim,
            expr_embedding_dim=self.expr_embedding_dim,
            expr_hidden_dim=512,
            expr_n_heads=8,
            expr_n_layers=3,
            node_feature_dim=len(KeyType) + 256,
            node_embedding_dim=512,
            node_hidden_dim=1024,
            node_n_heads=8,
            node_n_layers=2,
            plan_embedding_dim=512,
            plan_hidden_dim=1024,
            plan_n_heads=8,
            plan_n_layers=6,
            card_info_dim=256,
            cost_embedding_dim=192,
            cost_hidden_dim=256,
            cost_n_heads=8,
            cost_n_layers=1
        )
        self.model = CatModel(model_arg)

    def transform(self, plans: list[dict]) -> list[Sample]:
        samples: list[tuple[dict, dict[str, str], list[list[tuple[InputTokenType, any]]]]] = []

        for plan in plans:
            alias_map, rel_names, _ = get_alias_map(plan)
            exprs, _ = featurize_exprs(plan, 0, alias_map, rel_names)
            samples.append((plan, alias_map, exprs))

        ret = []
        for plan, alias_map, exprs in samples:
            expr_vecs: list[torch.Tensor] = []
            for expr in exprs:
                expr_vecs.append(self._vectorize_tokens(expr))
            ret.append(Sample(plan, alias_map, expr_vecs))

        return ret

    def transform_sample(self, sample: Sample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        expr_embedding_manager = EmbeddingManager(sample.exprs, self.model.expr_encoder)
        nodes, pos, mask_range, cards = featurize_nodes(sample.plan, self.expr_embedding_dim, sample.alias_map, self.table_map, self.column_map, expr_embedding_manager, device)
        nodes = batch_seqs(nodes)
        x = self.model.node_encoder(*nodes)
        pos = torch.tensor(pos, dtype=torch.float32, device=device)
        mask = torch.full((len(mask_range), len(mask_range)), -torch.inf, dtype=torch.float32, device=device)
        cards = torch.log(torch.tensor(cards, dtype=torch.float32, device=device) + 1.) / math.log(self.max_card + 1.)
        cost = sample.plan.get('Execution Time', torch.inf)
        for idx, (begin, end) in enumerate(mask_range):
            mask[idx, begin:end] = 0.
        return x, pos, mask, cards, cost

    def batch_transformed_samples(
            self,
            xs: list[torch.Tensor],
            positions: list[torch.Tensor],
            masks: list[torch.Tensor],
            cards_batch: list[torch.Tensor],
            costs: list[float],
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = len(xs)
        max_seq_len = max(x.shape[0] for x in xs)
        dim = xs[0].shape[1]
        ret_x = torch.zeros(batch_size, max_seq_len, dim, dtype=torch.float32, device=device)
        ret_pos = torch.zeros(batch_size, max_seq_len, 4, dtype=torch.float32, device=device)
        ret_mask = torch.zeros(batch_size, max_seq_len, max_seq_len, dtype=torch.float32, device=device)
        ret_cards = torch.full((batch_size, max_seq_len), -torch.inf, dtype=torch.float32, device=device)
        ret_cost = torch.tensor(costs, dtype=torch.float32, device=device)

        for idx, (x, pos, mask, cards) in enumerate(zip(xs, positions, masks, cards_batch)):
            seq_len = x.shape[0]
            ret_x[idx, :seq_len] = x
            ret_pos[idx, :seq_len] = pos
            ret_mask[idx, :seq_len, :seq_len] = mask
            ret_mask[idx, :seq_len, seq_len:] = -torch.inf
            ret_cards[idx, :seq_len] = cards
        return ret_x, ret_pos, ret_mask, ret_cards, ret_cost

    def transform_samples(self, batch: list[Sample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = []
        positions = []
        masks = []
        cards_batch = []
        costs = []

        for plan in batch:
            x, pos, mask, cards, cost = self.transform_sample(plan, device)
            xs.append(x)
            positions.append(pos)
            masks.append(mask)
            cards_batch.append(cards)
            costs.append(cost)

        return self.batch_transformed_samples(xs, positions, masks, cards_batch, costs, device)

    def get_collate_fn(self, device: torch.device):
        return lambda batch: self.transform_samples(batch, device)

    def get_collate_fn2(self, device: torch.device):
        return lambda batch: self.batch_transformed_samples(*zip(*batch), device)

    def denormalize_card(self, card: float) -> float:
        return torch.exp(card * math.log(self.max_card + 1.))

    def q_error_loss(self, input, target):
        input = self.denormalize_card(input)
        target = self.denormalize_card(target)
        max_v = torch.max(input, target)
        min_v = torch.min(input, target)
        return max_v / min_v

    def save(self, path: str) -> None:
        model_state = {
            'word_table': self.word_table,
            'max_table_idx': self.max_table_idx,
            'max_card': self.max_card,
            'feature_dim': self.feature_dim,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        model_state = torch.load(path)
        self.word_table = model_state['word_table']
        self.max_table_idx = model_state['max_table_idx']
        self.max_card = model_state['max_card']
        self.feature_dim = model_state['feature_dim']
        self.init_model()
        self.model.load_state_dict(model_state['state_dict'])

    def _fit_sample(self, sample: dict, alias_map: dict[str, tuple[str, int]], rel_names: set[str]) -> tuple[list[tuple[InputTokenType, any]], int]:
        root = sample['Plan']
        seqs = []

        def dfs(node, parent=None):
            if 'Hash Cond' in node:
                seqs.append(tokenizer(node['Hash Cond'], alias_map, rel_names, node, parent))
            if 'Index Cond' in node:
                seqs.append(tokenizer(node['Index Cond'], alias_map, rel_names, node, parent))
            if 'Recheck Cond' in node:
                seqs.append(tokenizer(node['Recheck Cond'], alias_map, rel_names, node, parent))
            if 'Merge Cond' in node:
                seqs.append(tokenizer(node['Merge Cond'], alias_map, rel_names, node, parent))
            if 'Filter' in node:
                seqs.append(tokenizer(node['Filter'], alias_map, rel_names, node, parent))
            if 'Join Filter' in node:
                seqs.append(tokenizer(node['Join Filter'], alias_map, rel_names, node, parent))
            for child in node.get('Plans', []):
                dfs(child, node)

        dfs(root)

        for tokens, max_table_idx in seqs:
            self.max_table_idx = max(self.max_table_idx, max_table_idx)
            for t, token in tokens:
                if t == InputTokenType.WORD:
                    if token not in self.word_table:
                        self.word_table[token] = len(self.word_table)

    def _normalize_number(self, key, value):
        min_v, max_v = self.normalizer[key]
        return (value - min_v) / (max_v - min_v)

    def _vectorize_tokens(self, tokens) -> torch.Tensor:
        seq = torch.zeros((len(tokens), self.feature_dim), dtype=torch.float)
        for idx, (t, token) in enumerate(tokens):
            seq[idx, t.value] = 1.
            match t:
                case InputTokenType.SYNTAX:
                    seq[idx, self.num_input_types + token.value] = 1.
                case InputTokenType.TABLE:
                    seq[idx, self.num_input_types + self.table_map[token]] = 1.
                case InputTokenType.TABLE_IDX:
                    seq[idx, self.num_input_types + token] = 1.
                case InputTokenType.COLUMN:
                    seq[idx, self.num_input_types + self.column_map[token]] = 1.
                case InputTokenType.WORD:
                    seq[idx, self.num_input_types + self.word_table[token]] = 1.
                case InputTokenType.NUMBER:
                    seq[idx, self.num_input_types] = self._normalize_number(*token)
        return seq
