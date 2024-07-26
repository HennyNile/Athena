import argparse
import copy
import os
import math
import re
import json

import torch
import torch.nn.functional as F
from mamba_ssm import BatchedTree2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np

from lero_model import LeroNet

SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Gather', 'Gather Merge', 'Bitmap Index Scan', 'Memoize']
OP_TYPES = ["Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES
OP_TYPES = SCAN_TYPES + JOIN_TYPES

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
    
    def __str__(self) -> str:
        # a tree representation
        def dfs(node: IndexTreeNode, is_last: list[bool]):
            indent = ''
            if len(is_last) > 0:
                for i in range(len(is_last) - 1):
                    indent += '    ' if is_last[i] else '│   '
                indent += '└── ' if is_last[-1] else '├── '
            result = indent + f'{node.idx}\n'
            for i, child in enumerate(node.children):
                result += dfs(child, is_last + [i == len(node.children) - 1])
            return result
        result = dfs(self, [])
        return result

class Tree:
    def __init__(self, left_first_x: np.ndarray, right_first_x: np.ndarray, left_first_indices: IndexTreeNode, right_first_indices: IndexTreeNode):
        self.left_first_x = left_first_x
        self.right_first_x = right_first_x
        self.left_first_indices = left_first_indices
        self.right_first_indices = right_first_indices

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

def batch_trees(trees: list[Tree]) -> BatchedTree2:
    batch_size = len(trees)
    max_len = max([tree.left_first_x.shape[0] for tree in trees])
    dim = trees[0].left_first_x.shape[1]
    x = np.zeros((2 * batch_size, max_len, dim), dtype=np.float32)
    conv_indices = np.zeros((2 * batch_size, 4 * max_len), dtype=np.int64)
    output_indices = np.zeros(2 * batch_size, dtype=np.int64)
    for i, tree in enumerate(trees):
        assert tree.left_first_x.shape[0] == tree.right_first_x.shape[0]
        seq_len = tree.left_first_x.shape[0]
        x[i * 2, :seq_len] = tree.left_first_x
        x[i * 2 + 1, :seq_len] = tree.right_first_x
        output_indices[i * 2] = tree.left_first_indices.idx
        output_indices[i * 2 + 1] = tree.right_first_indices.idx
        def left_first_dfs(node: IndexTreeNode):
            conv_indices[i * 2, node.idx * 4] = node.idx + 1
            if len(node) == 0:
                pass
            elif len(node) == 1:
                conv_indices[i * 2, node.idx * 4 + 1] = node.children[0].idx + 1
                left_first_dfs(node.children[0])
            else:
                assert len(node) <= 2
                conv_indices[i * 2, node.idx * 4 + 2] = node.children[0].children[0].idx + 1
                conv_indices[i * 2, node.idx * 4 + 3] = node.children[1].children[0].idx + 1
                left_first_dfs(node.children[0])
                left_first_dfs(node.children[1])
        def right_first_dfs(node: IndexTreeNode):
            conv_indices[i * 2 + 1, node.idx * 4] = node.idx + 1
            if len(node) == 0:
                pass
            elif len(node) == 1:
                conv_indices[i * 2 + 1, node.idx * 4 + 1] = node.children[0].idx + 1
                right_first_dfs(node.children[0])
            else:
                assert len(node) <= 2
                conv_indices[i * 2 + 1, node.idx * 4 + 3] = node.children[0].children[0].idx + 1
                conv_indices[i * 2 + 1, node.idx * 4 + 2] = node.children[1].children[0].idx + 1
                right_first_dfs(node.children[0])
                right_first_dfs(node.children[1])
        left_first_dfs(tree.left_first_indices)
        right_first_dfs(tree.right_first_indices)
    x = torch.tensor(x)
    conv_indices = torch.tensor(conv_indices)
    output_indices = torch.tensor(output_indices)
    return BatchedTree2(x, conv_indices, output_indices)

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
        self.lr_offset = 0
        self.op_offset = self.lr_offset + 2
        self.rel_offset = self.op_offset + len(OP_TYPES)
        self.lcond_offset = self.rel_offset + len(self.input_relations)
        self.rcond_offset = self.lcond_offset + len(self.input_relations)
        self.width_offset = self.rcond_offset + len(self.input_relations)
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
        left_first_vecs: list[np.ndarray] = []
        right_first_vecs: list[np.ndarray] = []
        node_idx = 0
        def left_first_dfs(node: dict) -> IndexTreeNode:
            if node['Node Type'] not in OP_TYPES:
                return left_first_dfs(node['Plans'][0])
            nonlocal node_idx

            childrens = []
            if node['Node Type'] in JOIN_TYPES:
                for i, plan in enumerate(node['Plans']):
                    child_idx_node = left_first_dfs(plan)
                    idx_node = IndexTreeNode(node_idx)
                    node_idx += 1
                    idx_node.append(child_idx_node)
                    childrens.append(idx_node)
                    pos_vec = np.zeros(self.input_feature_dim, dtype=np.float32)
                    pos_vec[self.lr_offset + i] = 1.
                    left_first_vecs.append(pos_vec)
            idx_node = IndexTreeNode(node_idx)
            node_idx += 1
            for child in childrens:
                idx_node.append(child)
            vec = self._transform_node(node, plan_info)
            left_first_vecs.append(vec)

            return idx_node
        
        def right_first_dfs(node: dict) -> IndexTreeNode:
            if node['Node Type'] not in OP_TYPES:
                return right_first_dfs(node['Plans'][-1])
            nonlocal node_idx

            childrens = []
            if node['Node Type'] in JOIN_TYPES:
                for i, plan in enumerate(reversed(node['Plans'])):
                    child_idx_node = right_first_dfs(plan)
                    idx_node = IndexTreeNode(node_idx)
                    node_idx += 1
                    idx_node.append(child_idx_node)
                    childrens.append(idx_node)
                    pos_vec = np.zeros(self.input_feature_dim, dtype=np.float32)
                    pos_vec[self.lr_offset + 1 - i] = 1.
                    right_first_vecs.append(pos_vec)
            idx_node = IndexTreeNode(node_idx)
            node_idx += 1
            for child in childrens:
                idx_node.append(child)
            vec = self._transform_node(node, plan_info)
            right_first_vecs.append(vec)

            return idx_node

        left_first_idx = left_first_dfs(plan['Plan'])
        node_idx = 0
        right_first_idx = right_first_dfs(plan['Plan'])
        result = LeroSample(Tree(np.stack(left_first_vecs), np.stack(right_first_vecs), left_first_idx, right_first_idx), plan)
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
