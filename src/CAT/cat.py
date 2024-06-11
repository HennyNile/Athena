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

table_name_regex = re.compile(r'^CREATE TABLE ([\w_]+) .*')
column_name_regex = re.compile(r'^\s*([\w_]+) (integer|character).*')

def get_column_map(path) -> tuple[dict[str, int], dict[str, int]]:
    with open(path, 'r') as f:
        lines = f.readlines()
    table_map = {}
    column_map = {}
    for line in lines:
        match_result = table_name_regex.match(line)
        if match_result:
            current_table_name = match_result.group(1)
            table_idx = len(table_map)
            table_map[current_table_name] = table_idx
            continue
        match_result = column_name_regex.match(line)
        if match_result:
            column_name = match_result.group(1)
            column_idx = len(column_map)
            column_map[(current_table_name, column_name)] = column_idx
    return table_map, column_map

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
    def __init__(self, table_map, column_map, normalizer):
        self.table_map = table_map
        self.column_map = column_map
        self.normalizer = normalizer
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

    def transform_dataset(self, dataset: list[list[dict]]) -> list[Sample]:
        all_samples = []
        for plans in dataset:
            samples = self.transform(plans)
            all_samples.extend(samples)
        return all_samples

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
        costs = sample.plan.get('Execution Time', torch.inf)
        for idx, (begin, end) in enumerate(mask_range):
            mask[idx, begin:end] = 0.
        return x, pos, mask, cards, costs

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

        batch_size = len(xs)
        max_seq_len = max(x.shape[0] for x in xs)
        dim = xs[0].shape[1]
        ret_x = torch.zeros(batch_size, max_seq_len, dim, dtype=torch.float32, device=device)
        ret_pos = torch.zeros(batch_size, max_seq_len, 4, dtype=torch.float32, device=device)
        ret_mask = torch.zeros(batch_size, max_seq_len, max_seq_len, dtype=torch.float32, device=device)
        ret_cards = torch.zeros(batch_size, max_seq_len, dtype=torch.float32, device=device)
        ret_cost = torch.tensor(costs, dtype=torch.float32, device=device)

        for idx, (x, pos, mask, cards) in enumerate(zip(xs, positions, masks, cards_batch)):
            seq_len = x.shape[0]
            ret_x[idx, :seq_len] = x
            ret_pos[idx, :seq_len] = pos
            ret_mask[idx, :seq_len, :seq_len] = mask
            ret_mask[idx, :seq_len, seq_len:] = -torch.inf
            ret_cards[idx, :seq_len] = cards
        return ret_x, ret_pos, ret_mask, ret_cards, ret_cost
    
    def get_collate_fn(self, device: torch.device):
        return lambda batch: self.transform_samples(batch, device)

    def denormalize_card(self, card: float) -> float:
        return torch.exp(card * math.log(self.max_card + 1.))

    def q_error_loss(self, input, target):
        input = self.denormalize_card(input)
        target = self.denormalize_card(target)
        max_v = torch.max(input, target)
        min_v = torch.min(input, target)
        return max_v / min_v

    def itemwise_loss(self, samples: list[Sample], train_cost = False) -> list[torch.Tensor]:
        x, pos, mask, cards_batch, costs = self.transform_samples(samples)
        costs_output, cards_output = self.model.train_output(x, pos, mask)
        cards_mask = mask[:, 0].isfinite() & cards_batch.isfinite()
        cards_output_vec = cards_output[cards_mask]
        cards_label_vec = cards_batch[cards_mask]
        weight = cards_label_vec / torch.sum(cards_label_vec)
        weight = weight.clamp(min=1e-3)
        cards_loss = torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
        if self.args.output_mode == 'number':
            costs_loss = F.mse_loss(costs_output.view(-1), costs / 10000. - 12)
        elif self.args.output_mode == 'ce':
            costs_loss = F.cross_entropy(costs_output.view(-1, 2), costs.view(-1))
        elif self.args.output_mode == 'bce':
            costs_loss = F.binary_cross_entropy_with_logits(costs_output, costs)
        else:
            raise ValueError(f'Invalid output_mode: {self.args.output_mode}')
        if train_cost:
            return [costs_loss]
        else:
            return [cards_loss]

    def pairwise_loss(self, samples: list[Sample], train_cost = False) -> list[torch.Tensor]:
        x, pos, mask, cards_batch, costs = self.transform_samples(samples)
        costs_output, cards_output = self.model.train_output(x, pos, mask)
        cards_mask = mask[:, 0].isfinite() & cards_batch.isfinite()
        cards_output_vec = cards_output[cards_mask]
        cards_label_vec = cards_batch[cards_mask]
        weight = cards_label_vec / torch.sum(cards_label_vec)
        weight = weight.clamp(min=1e-3)
        cards_loss = torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
        # cards_loss = torch.mean(self.q_error_loss(torch.clamp(cards_output_vec, 0., 1.2), cards_label_vec))
        # cards_loss = F.smooth_l1_loss(cards_output_vec, cards_label_vec)
        if self.args.output_mode == 'number':
            output_diffs = costs_output - costs_output.view(-1)
            # costs = torch.log(costs)
            # label_diffs = F.sigmoid((costs.unsqueeze(1) - costs) * 10)
            label_diffs = ((costs.unsqueeze(1) - costs) > 0.).float()
            mask = torch.tril(torch.ones_like(output_diffs, dtype=torch.bool), diagonal=-1)
            mask = mask & label_diffs.isfinite()
            output_diffs = output_diffs[mask]
            label_diffs = label_diffs[mask]
            cost_loss = F.binary_cross_entropy_with_logits(output_diffs, label_diffs)
        else:
            raise ValueError(f'Invalid output_mode: {self.args.output_mode}')
        if train_cost:
            return [cost_loss]
        else:
            return [cards_loss]

    def listwise_loss(self, samples: list[Sample]) -> list[torch.Tensor]:
        x, pos, mask, cards_batch, costs = self.transform_samples(samples)
        costs_output, cards_output = self.model.train_output(x, pos, mask)
        cards_mask = mask[:, 0].isfinite() & cards_batch.isfinite()
        cards_output_vec = cards_output[cards_mask]
        cards_label_vec = cards_batch[cards_mask]
        cards_loss = F.mse_loss(cards_output_vec, cards_label_vec)
        if self.args.output_mode == 'number':
            costs_loss = lambda_loss(-costs_output.view(-1), costs)
        else:
            raise ValueError(f'Invalid output_mode: {self.args.output_mode}')
        return [cards_loss + 10 * costs_loss]

    def optimizers(self, train_cost = False) -> list[torch.optim.Optimizer]:
        # common_parameters = [self.model.expr_encoder.parameters(), self.model.node_encoder.parameters(), self.model.plan_encoder.parameters()]
        # card_parameters = common_parameters + [self.model.card_predictor.parameters(), self.model.card_estimator.parameters()]
        # cost_parameters = [self.model.cost_predictor.parameters(), self.model.cost_estimator.parameters()]
        # card_optimizer = torch.optim.Adam(itertools.chain(*card_parameters))
        # cost_optimizer = torch.optim.Adam(itertools.chain(*cost_parameters))
        # return [card_optimizer, cost_optimizer]
        if self.args.freeze_pretrain:
            self.model.expr_encoder.requires_grad_(False)
            self.model.node_encoder.requires_grad_(False)
            self.model.plan_encoder.requires_grad_(False)
            self.model.plan_encoder.blocks[-1].requires_grad_(True)
            net_modules = [self.model.plan_encoder.blocks[-1], self.model.card_predictor, self.model.card_estimator, self.model.cost_predictor, self.model.cost_estimator, self.model.batch_norm]
            parameters = [m.parameters() for m in net_modules]
            return [torch.optim.Adam(itertools.chain(*parameters), lr=1e-5)]
        else:
            if train_cost:
                return [torch.optim.Adam(self.model.parameters(), lr=1e-5)]
            else:
                return [torch.optim.Adam(self.model.parameters(), lr=1e-6)]

    def lr_scheduler(self, optimizers: list[torch.optim.Optimizer], train_cost = False) -> list[torch.optim.lr_scheduler.LRScheduler]:
        if train_cost:
            # return [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92) for optimizer in optimizers]
            return [torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.6) for optimizer in optimizers]
        else:
            return [None for optimizer in optimizers]

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
