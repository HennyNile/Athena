import argparse
import os
import math
import re
import json

import torch
import torch.nn.functional as F
from mamba_ssm import BatchedTree
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np

from lero_model import LeroNet

SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Gather', 'Gather Merge', 'Bitmap Index Scan', 'Memoize']
OP_TYPES = ["Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES
# OP_TYPES = SCAN_TYPES + JOIN_TYPES

class IndexTreeNode:
    def __init__(self, idx: int):
        self.idx = idx
        self.children: list[IndexTreeNode] = []

    def append(self, idx: 'int|IndexTreeNode'):
        if type(idx) == IndexTreeNode:
            new_node = idx
        else:
            new_node = IndexTreeNode(idx)
        self.children.append(new_node)

    def __getitem__(self, i: int) -> 'IndexTreeNode':
        return self.children[i]
    
    def __len__(self) -> int:
        return len(self.children)

class Tree:
    def __init__(self, x: np.ndarray, indices: IndexTreeNode):
        self.x = x
        self.indices = indices

class LeroSample:
    def __init__(self, tree: Tree, plan: dict):
        self.tree = tree
        self.plan = plan

class PlanInfo:
    def __init__(self, alias_map: dict[str, tuple[str, int]], rel_names: set[str]) -> None:
        self.alias_map = alias_map
        self.rel_names = rel_names

def get_alias_map(sample) -> tuple[PlanInfo, int]:
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
    return PlanInfo(alias_map, relation_names), max_card

def batch_trees(trees: list[Tree]) -> BatchedTree:
    batch_size = len(trees)
    max_len = max([tree.x.shape[0] for tree in trees])
    dim = trees[0].x.shape[1]
    x = np.zeros((batch_size, max_len, dim), dtype=np.float32)
    conv_indices = np.zeros((batch_size, 4 * max_len), dtype=np.int64)
    indices_list: list[list[tuple[int, int]]] = []
    state_indices: list[list[tuple[int, int]]] = []
    for i, tree in enumerate(trees):
        x[i, :tree.x.shape[0]] = tree.x
        def dfs(node: IndexTreeNode, level: int = 0):
            conv_indices[i, node.idx * 4] = node.idx + 1
            if level >= len(indices_list):
                indices_list.append([])
                state_indices.append([])
            indices_list[level].append((i, node.idx))
            state_indices[level].append((0, 0))
            level_index = len(state_indices[level])
            if len(node) == 0:
                pass
            elif len(node) == 1:
                conv_indices[i, node.idx * 4 + 1] = node.children[0].idx + 1
                child_level_index = dfs(node.children[0], level + 1)
                state_indices[level][level_index - 1] = (child_level_index, child_level_index)
            else:
                assert len(node) <= 2
                conv_indices[i, node.idx * 4 + 2] = node.children[0].idx + 1
                conv_indices[i, node.idx * 4 + 3] = node.children[1].idx + 1
                child_level_index1 = dfs(node.children[0], level + 1)
                child_level_index2 = dfs(node.children[1], level + 1)
                state_indices[level][level_index - 1] = (child_level_index1, child_level_index2)
            return level_index
        dfs(tree.indices)
    indices_list = [np.array(l, dtype=np.int64) for l in indices_list]
    state_indices = [np.array(l, dtype=np.int64) for l in state_indices]
    x = torch.tensor(x)
    indices_list = [torch.tensor(l) for l in indices_list]
    state_indices = [torch.tensor(l) for l in state_indices]
    conv_indices = torch.tensor(conv_indices)
    return BatchedTree(x, indices_list, state_indices, conv_indices)

class Lero:
    def __init__(self, table_map):
        self.input_relations = table_map
        self.min_est_card = float('inf')
        self.max_est_card = 0.
        self.min_width = float('inf')
        self.max_width = 0.
        self.model: LeroNet = None

    def fit_train(self, dataset: list[list[dict]]) -> None:
        plans = [sample for query in dataset for sample in query]
        def dfs(node: dict) -> None:
            est_card = node['Plan Rows']
            self.min_est_card = min(self.min_est_card, est_card)
            self.max_est_card = max(self.max_est_card, est_card)

            width = node['Plan Width']
            self.min_width = min(self.min_width, width)
            self.max_width = max(self.max_width, width)

            for child in node.get('Plans', []):
                dfs(child)

        for plan in plans:
            dfs(plan['Plan'])

        self.min_est_card = math.log(self.min_est_card + 1)
        self.max_est_card = math.log(self.max_est_card + 1)

    def init_model(self) -> None:
        self.op_offset = 0
        self.rel_offset = self.op_offset + len(OP_TYPES)
        self.width_offset = self.rel_offset + len(self.input_relations)
        self.rows_offset = self.width_offset + 1
        self.input_feature_dim = self.rows_offset + 1
        self.model = LeroNet(self.input_feature_dim)

    def transform(self, plans: list[dict]) -> list[LeroSample]:
        samples = []
        for plan in plans:
            samples.append(self._transform_plan(plan))
        return samples

    def save(self, path: str) -> None:
        model_state = {
            'min_card': self.min_est_card,
            'max_card': self.max_est_card,
            'min_width': self.min_width,
            'max_width': self.max_width,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        model_state = torch.load(path)
        self.min_est_card = model_state['min_card']
        self.max_est_card = model_state['max_card']
        self.min_width = model_state['min_width']
        self.max_width = model_state['max_width']
        self.init_model()
        self.model.load_state_dict(model_state['state_dict'])

    def _transform_samples(self, samples: list[LeroSample]):
        times = [sample.plan.get('Execution Time', torch.inf) for sample in samples]
        weights = [sample.plan['Execution Time'] if 'Execution Time' in sample.plan else sample.plan['Timeout Time'] for sample in samples]
        times = torch.tensor(times, dtype=torch.float32, device=torch.device('cuda'))
        weights = torch.tensor(weights, dtype=torch.float32, device=torch.device('cuda'))
        trees = batch_trees([sample.tree for sample in samples])
        return trees, times, weights

    def _norm_est_card(self, est_card: float) -> float:
        return (math.log(est_card + 1) - self.min_est_card) / (self.max_est_card - self.min_est_card)
    
    def _norm_width(self, width: int) -> float:
        return width / self.max_width

    def _transform_plan(self, plan: dict) -> LeroSample:
        plan_info, _ = get_alias_map(plan)
        vecs: list[np.ndarray] = []
        node_idx = 0
        def dfs(node: dict) -> IndexTreeNode:
            # if node['Node Type'] not in OP_TYPES:
            #     return dfs(node['Plans'][0])
            nonlocal node_idx
            vec = self._transform_node(node, plan_info)
            vecs.append(vec)
            idx_node = IndexTreeNode(node_idx)
            node_idx += 1

            # if node['Node Type'] in JOIN_TYPES:
            for plan in node.get('Plans', []):
                child_idx_node = dfs(plan)
                idx_node.append(child_idx_node)

            return idx_node

        root_idx = dfs(plan['Plan'])
        result = LeroSample(Tree(np.stack(vecs), root_idx), plan)
        return result

    cond_pattern = re.compile(r'\(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\)')
    cond_pattern2 = re.compile(r'\(\(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\) AND \(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\)\)')
    cond_pattern3 = re.compile(r'\(\(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\) AND \(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\) AND \(([a-zA-Z0-9_]+).([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\)\)')
    index_cond_pattern = re.compile(r'\(([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+).([a-zA-Z0-9_]+)\)')

    def _transform_node(self, node: dict, plan_info: PlanInfo) -> np.ndarray:
        op_type = node['Node Type']
        arr = np.zeros(self.input_feature_dim, dtype=np.float32)
        arr[self.op_offset + OP_TYPES.index(op_type)] = 1.
        have_cond = False
        if node['Node Type'] in SCAN_TYPES:
            arr[self.rel_offset + self.input_relations[node['Relation Name']]] = 1.
            # if node['Node Type'] in ['Index Scan', 'Index Only Scan'] and 'Index Cond' in node:
            #     have_cond = True
            #     index_cond = node['Index Cond']
            #     l_rel = node['Relation Name']
            #     _, r_alias, _ = self.index_cond_pattern.match(index_cond).groups()
            #     r_rel, _ = plan_info.alias_map[r_alias]
        elif node['Node Type'] in JOIN_TYPES:
            if node['Node Type'] == 'Hash Join':
                have_cond = True
                join_cond = node['Hash Cond']
                try:
                    l_alias, _, r_alias, _ = self.cond_pattern.match(join_cond).groups()
                except Exception as e:
                    try:
                        l_alias, _, r_alias, _, l2_alias, _, r2_alias, _ = self.cond_pattern2.match(join_cond).groups()
                        l2_rel, _ = plan_info.alias_map[l2_alias]
                        r2_rel, _ = plan_info.alias_map[r2_alias]
                        arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                        arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                    except Exception as e:
                        try:
                            l_alias, _, r_alias, _, l2_alias, _, r2_alias, _, l3_alias, _, r3_alias, _ = self.cond_pattern3.match(join_cond).groups()
                            l2_rel, _ = plan_info.alias_map[l2_alias]
                            r2_rel, _ = plan_info.alias_map[r2_alias]
                            l3_rel, _ = plan_info.alias_map[l3_alias]
                            r3_rel, _ = plan_info.alias_map[r3_alias]
                            arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                            arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                            arr[self.rel_offset + self.input_relations[l3_rel]] = 1
                            arr[self.rel_offset + self.input_relations[r3_rel]] = 1
                        except Exception as e:
                            print(join_cond)
                            raise e
                l_rel, _ = plan_info.alias_map[l_alias]
                r_rel, _ = plan_info.alias_map[r_alias]
            elif node['Node Type'] == 'Merge Join':
                have_cond = True
                join_cond = node['Merge Cond']
                try:
                    l_alias, _, r_alias, _ = self.cond_pattern.match(join_cond).groups()
                except Exception as e:
                    try:
                        l_alias, _, r_alias, _, l2_alias, _, r2_alias, _ = self.cond_pattern2.match(join_cond).groups()
                        l2_rel, _ = plan_info.alias_map[l2_alias]
                        r2_rel, _ = plan_info.alias_map[r2_alias]
                        arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                        arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                    except Exception as e:
                        try:
                            l_alias, _, r_alias, _, l2_alias, _, r2_alias, _, l3_alias, _, r3_alias, _ = self.cond_pattern3.match(join_cond).groups()
                            l2_rel, _ = plan_info.alias_map[l2_alias]
                            r2_rel, _ = plan_info.alias_map[r2_alias]
                            l3_rel, _ = plan_info.alias_map[l3_alias]
                            r3_rel, _ = plan_info.alias_map[r3_alias]
                            arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                            arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                            arr[self.rel_offset + self.input_relations[l3_rel]] = 1
                            arr[self.rel_offset + self.input_relations[r3_rel]] = 1
                        except Exception as e:
                            print(join_cond)
                            raise e
                l_rel, _ = plan_info.alias_map[l_alias]
                r_rel, _ = plan_info.alias_map[r_alias]
            elif node['Node Type'] == 'Nested Loop':
                if 'Join Filter' in node:
                    have_cond = True
                    join_cond = node['Join Filter']
                    try:
                        l_alias, _, r_alias, _ = self.cond_pattern.match(join_cond).groups()
                    except Exception as e:
                        try:
                            l_alias, _, r_alias, _, l2_alias, _, r2_alias, _ = self.cond_pattern2.match(join_cond).groups()
                            l2_rel, _ = plan_info.alias_map[l2_alias]
                            r2_rel, _ = plan_info.alias_map[r2_alias]
                            arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                            arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                        except Exception as e:
                            try:
                                l_alias, _, r_alias, _, l2_alias, _, r2_alias, _, l3_alias, _, r3_alias, _ = self.cond_pattern3.match(join_cond).groups()
                                l2_rel, _ = plan_info.alias_map[l2_alias]
                                r2_rel, _ = plan_info.alias_map[r2_alias]
                                l3_rel, _ = plan_info.alias_map[l3_alias]
                                r3_rel, _ = plan_info.alias_map[r3_alias]
                                arr[self.rel_offset + self.input_relations[l2_rel]] = 1
                                arr[self.rel_offset + self.input_relations[r2_rel]] = 1
                                arr[self.rel_offset + self.input_relations[l3_rel]] = 1
                                arr[self.rel_offset + self.input_relations[r3_rel]] = 1
                            except Exception as e:
                                print(join_cond)
                                raise e
                    l_rel, _ = plan_info.alias_map[l_alias]
                    r_rel, _ = plan_info.alias_map[r_alias]
                else:
                    right_child = node['Plans'][1]
                    try:
                        while right_child['Node Type'] not in ['Index Scan', 'Index Only Scan']:
                            if 'Plans' not in right_child or len(right_child['Plans']) != 1:
                                raise RuntimeError(f'Unexpected right child')
                            right_child = right_child['Plans'][0]
                        have_cond = True
                    except:
                        pass
                    if have_cond:
                        l_rel = right_child['Relation Name']
                        _, r_alias, _ = self.index_cond_pattern.match(right_child['Index Cond']).groups()
                        r_rel, _ = plan_info.alias_map[r_alias]
        if have_cond:
            arr[self.rel_offset + self.input_relations[l_rel]] = 1
            arr[self.rel_offset + self.input_relations[r_rel]] = 1
        arr[self.width_offset] = node['Plan Width']
        # arr[self.width_offset] = self._norm_width(node['Plan Width'])
        arr[self.rows_offset] = self._norm_est_card(node['Plan Rows'])
        return arr
