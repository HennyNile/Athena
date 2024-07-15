import argparse
import os
import math
import re

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np

from TreeConvolution.util import prepare_trees
from lero_model import LeroNet

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES

class LeroSample:
    def __init__(self, feature: np.ndarray):
        self.feature = feature
        self.plan = None
        self.left = None
        self.right = None

class Lero:
    def __init__(self, table_map):
        self.input_relations = table_map
        self.input_feature_dim = len(OP_TYPES) + len(self.input_relations) + 2
        self.rel_offset = len(OP_TYPES)
        self.width_offset = self.rel_offset + len(self.input_relations)
        self.rows_offset = self.width_offset + 1
        self.min_est_card = float('inf')
        self.max_est_card = 0.
        self.max_width = 0.
        self.model: torch.nn.Module = None

    def fit_train(self, dataset: list[list[dict]]) -> None:
        plans = [sample for query in dataset for sample in query]
        def dfs(node: dict) -> None:
            est_card = node['Plan Rows']
            self.min_est_card = min(self.min_est_card, est_card)
            self.max_est_card = max(self.max_est_card, est_card)

            width = node['Plan Width']
            self.max_width = max(self.max_width, width)

            for child in node.get('Plans', []):
                dfs(child)
        
        for plan in plans:
            dfs(plan['Plan'])

        self.min_est_card = math.log(self.min_est_card + 1)
        self.max_est_card = math.log(self.max_est_card + 1)

    def init_model(self) -> None:
        self.model = LeroNet(self.input_feature_dim)

    def transform(self, plans: list[dict]):
        samples = []
        for plan in plans:
            samples.append(self._transform_plan(plan))
        return samples

    def save(self, path: str) -> None:
        model_state = {
            'min_card': self.min_est_card,
            'max_card': self.max_est_card,
            'max_width': self.max_width,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        model_state = torch.load(path)
        self.min_est_card = model_state['min_card']
        self.max_est_card = model_state['max_card']
        self.max_width = model_state['max_width']
        self.init_model()
        self.model.load_state_dict(model_state['state_dict'])

    def _transform_samples(self, samples: list[LeroSample]):
        times = [sample.plan.get('Execution Time', torch.inf) for sample in samples]
        weights = [sample.plan['Execution Time'] if 'Execution Time' in sample.plan else sample.plan['Timeout Time'] for sample in samples]
        times = torch.tensor(times, dtype=torch.float32, device=torch.device('cuda'))
        weights = torch.tensor(weights, dtype=torch.float32, device=torch.device('cuda'))
        trees = prepare_trees(samples, lambda x: x.feature, lambda x: x.left, lambda x: x.right, True, torch.device('cuda'))
        return trees, times, weights

    def _norm_est_card(self, est_card: float) -> float:
        return (math.log(est_card + 1) - self.min_est_card) / (self.max_est_card - self.min_est_card)
    
    def _norm_width(self, width: int) -> float:
        return width / self.max_width

    def _pad_sample(self) -> LeroSample:
        feature = np.zeros(self.input_feature_dim, dtype=np.float32)
        feature[0] = 1.
        return LeroSample(feature)

    def _transform_plan(self, plan: dict) -> LeroSample:
        def dfs(node: dict) -> tuple[LeroSample, list[str]]:
            children = []
            plan_relations = []

            for plan in node.get('Plans', []):
                child, subplan_relations = dfs(plan)
                children.append(child)
                plan_relations.extend(subplan_relations)

            if node['Node Type'] in SCAN_TYPES:
                plan_relations.append(node['Relation Name'])

            node_feature = self._transform_node(node, plan_relations)
            sample = LeroSample(node_feature)

            if len(children) == 0:
                pass
            elif len(children) == 1:
                sample.left = children[0]
                sample.right = self._pad_sample()
            elif len(children) == 2:
                sample.left = children[0]
                sample.right = children[1]
            else:
                raise RuntimeError('Number of children > 2')

            return sample, plan_relations

        result, _ = dfs(plan['Plan'])
        result.plan = plan
        return result

    def _transform_node(self, node: dict, relations: list[str]) -> np.ndarray:
        op_type = node['Node Type']
        arr = np.zeros(self.input_feature_dim, dtype=np.float32)
        if op_type in OP_TYPES:
            arr[OP_TYPES.index(op_type)] = 1.
        else:
            arr[0] = 1
        for rel in relations:
            arr[self.rel_offset + self.input_relations[rel]] += 1.
        arr[self.width_offset] = node['Plan Width']
        # arr[self.width_offset] = self._norm_width(node['Plan Width'])
        arr[self.rows_offset] = self._norm_est_card(node['Plan Rows'])
        return arr
