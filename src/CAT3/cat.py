import math
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np

sys.path.append('.')
from src.utils.db_utils import DataType, DBInfo
from src.utils.TreeConvolution.util import prepare_trees

from model import LeroNet
from expr_featurizer import PlanInfo, ExprParser, OpType, AtomicNode, Column, Number, Array, Null, ExprTree

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES

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

def batch_indices(batch: list[list[list[int]]]):
    max_len = max([len(l) for l in batch])
    ret: list[np.ndarray] = []
    for i in range(max_len):
        batch_i = []
        for l in batch:
            if i < len(l):
                batch_i.append(l[i])
            else:
                batch_i.append([1, 0])
        max_num_indices = max([len(l) for l in batch_i])
        for l in batch_i:
            l.extend([0] * (max_num_indices - len(l)))
        batch_i = np.array(batch_i, dtype=np.int64)
        ret.append(batch_i)
    return ret

class LeroSample:
    def __init__(self, feature: np.ndarray, cond_expr: ExprTree|None, filter_expr: ExprTree|None):
        self.feature = feature
        self.cond_expr = cond_expr
        self.filter_expr = filter_expr
        self.plan = None
        self.left = None
        self.right = None

class Lero:
    def __init__(self, db_info: DBInfo):
        self.db_info = db_info
        self.input_relations = db_info.table_map
        self.input_feature_dim = len(OP_TYPES) + len(self.input_relations) + 2
        self.rel_offset = len(OP_TYPES)
        self.width_offset = self.rel_offset + len(self.input_relations)
        self.rows_offset = self.width_offset + 1
        self.min_est_card = float('inf')
        self.max_est_card = 0.
        self.max_width = 0.
        self.max_alias_idx = 0
        self.max_array_len = 1
        self.model: LeroNet = None

    def _fit_sample(self, sample: dict, plan_info: PlanInfo) -> None:
        root = sample['Plan']

        def dfs(node, parent=None):
            if 'Hash Cond' in node:
                parser = ExprParser(node['Hash Cond'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            if 'Index Cond' in node:
                parser = ExprParser(node['Index Cond'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            if 'Recheck Cond' in node:
                parser = ExprParser(node['Recheck Cond'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            if 'Merge Cond' in node:
                parser = ExprParser(node['Merge Cond'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            if 'Filter' in node:
                parser = ExprParser(node['Filter'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            if 'Join Filter' in node:
                parser = ExprParser(node['Join Filter'], plan_info, self.db_info, node, parent)
                parser()
                self.max_alias_idx = max(self.max_alias_idx, parser.max_alias_idx)
                self.max_array_len = max(self.max_array_len, parser.max_array_len)
            for child in node.get('Plans', []):
                dfs(child, node)

        dfs(root)

    def fit_all(self, dataset: list[list[dict]]) -> None:
        # word_table and max_alias_idx
        plans = [sample for query in dataset for sample in query]
        for sample in plans:
            plan_info, _ = get_alias_map(sample)
            self._fit_sample(sample, plan_info)

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
        self.data_type_offset    = 0
        self.left_table_offset   = self.data_type_offset    + len(DataType)
        self.left_column_offset  = self.left_table_offset   + len(self.db_info.table_map)
        self.op_offset           = self.left_column_offset  + len(self.db_info.column_map)
        self.right_table_offset  = self.op_offset           + (2 + len(OpType))
        self.right_column_offset = self.right_table_offset  + len(self.db_info.table_map)
        self.number_offset       = self.right_column_offset + len(self.db_info.column_map)
        self.array_offset        = self.number_offset       + 1
        self.null_offset         = self.array_offset        + 1
        self.expr_dim            = self.null_offset         + 1

        self.model = LeroNet(self.input_feature_dim, self.expr_dim)

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
            'max_alias_idx': self.max_alias_idx,
            'max_array_len': self.max_array_len,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        model_state = torch.load(path)
        self.min_est_card = model_state['min_card']
        self.max_est_card = model_state['max_card']
        self.max_width = model_state['max_width']
        self.max_alias_idx = model_state['max_alias_idx']
        self.max_array_len = model_state['max_array_len']
        self.init_model()
        self.model.load_state_dict(model_state['state_dict'])

    def _transform_samples(self, samples: list[LeroSample]):
        times = [sample.plan.get('Execution Time', torch.inf) for sample in samples]
        times = torch.tensor(times, dtype=torch.float32, device=torch.device('cuda'))
        trees = prepare_trees(samples, lambda x: x.feature, lambda x: x.left, lambda x: x.right, True, torch.device('cuda'))
        _, _, max_nodes = trees[0].shape
        expr_list: list[np.ndarray] = []
        indices_list: list[list[list[int]]] = []
        cond_to_node: list[list[int]] = []
        filter_to_node: list[list[int]] = []
        for sample in samples:
            cond_indices: list[int] = [0]
            filter_indices: list[int] = [0]
            def dfs(sample_node: LeroSample):
                cond_index = 0
                filter_index = 0
                if sample_node.cond_expr is not None:
                    expr_list.append(sample_node.cond_expr.vecs)
                    indices_list.append(sample_node.cond_expr.indices)
                    cond_index = len(expr_list)
                if sample_node.filter_expr is not None:
                    expr_list.append(sample_node.filter_expr.vecs)
                    indices_list.append(sample_node.filter_expr.indices)
                    filter_index = len(expr_list)
                cond_indices.append(cond_index)
                filter_indices.append(filter_index)
                if sample_node.left is not None:
                    dfs(sample_node.left)
                if sample_node.right is not None:
                    dfs(sample_node.right)
            dfs(sample)
            cond_to_node.append(cond_indices)
            filter_to_node.append(filter_indices)
        num_trees = len(expr_list)
        max_exprs = max([l.shape[0] for l in expr_list])
        exprs = np.zeros((num_trees, max_exprs, self.expr_dim), dtype=np.float32)
        for i, expr in enumerate(expr_list):
            num_expr, _ = expr.shape
            exprs[i, :num_expr] = expr
        exprs = torch.tensor(exprs, device=torch.device('cuda'))
        indices = batch_indices(indices_list)
        indices = [torch.tensor(l, device=torch.device('cuda')) for l in indices]
        conds = []
        filters = []
        for c, f in zip(cond_to_node, filter_to_node):
            conds.extend(c)
            filters.extend(f)
            paddings = [0] * (max_nodes - len(c))
            conds.extend(paddings)
            filters.extend(paddings)
        conds = np.array(conds, dtype=np.int64)
        filters = np.array(filters, dtype=np.int64)
        conds = torch.tensor(conds, device=torch.device('cuda'))
        filters = torch.tensor(filters, device=torch.device('cuda'))
        return trees, times, exprs, indices, conds, filters

    def _norm_est_card(self, est_card: float) -> float:
        return (math.log(est_card + 1) - self.min_est_card) / (self.max_est_card - self.min_est_card)
    
    def _norm_width(self, width: int) -> float:
        return width / self.max_width

    def _pad_sample(self) -> LeroSample:
        feature = np.zeros(self.input_feature_dim, dtype=np.float32)
        feature[0] = 1.
        return LeroSample(feature, None, None)

    def _transform_plan(self, plan: dict) -> LeroSample:
        plan_info, _ = get_alias_map(plan)
        def dfs(node: dict, parent: dict|None = None) -> tuple[LeroSample, list[str]]:
            cond_expr = None
            if 'Hash Cond' in node:
                cond_expr = node['Hash Cond']
            elif 'Index Cond' in node:
                cond_expr = node['Index Cond']
            elif 'Recheck Cond' in node:
                cond_expr = node['Recheck Cond']
            elif 'Merge Cond' in node:
                cond_expr = node['Merge Cond']
            if cond_expr is not None:
                parser = ExprParser(cond_expr, plan_info, self.db_info, node, parent)
                cond_expr = parser()
                cond_expr.combine_disjunction()
                cond_expr.cache_features(self._featurize_atomic_expr)
            filter_expr = None
            if 'Filter' in node:
                filter_expr = node['Filter']
            elif 'Join Filter' in node:
                filter_expr = node['Join Filter']
            if filter_expr is not None:
                parser = ExprParser(filter_expr, plan_info, self.db_info, node, parent)
                filter_expr = parser()
                filter_expr.combine_disjunction()
                filter_expr.cache_features(self._featurize_atomic_expr)
            children = []
            plan_relations = []

            for plan in node.get('Plans', []):
                child, subplan_relations = dfs(plan, node)
                children.append(child)
                plan_relations.extend(subplan_relations)

            if node['Node Type'] in SCAN_TYPES:
                plan_relations.append(node['Relation Name'])

            node_feature = self._transform_node(node, plan_relations)
            sample = LeroSample(node_feature, cond_expr, filter_expr)

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
        arr[self.width_offset] = self._norm_width(node['Plan Width'])
        arr[self.rows_offset] = self._norm_est_card(node['Plan Rows'])
        return arr

    def _featurize_atomic_expr(self, node: AtomicNode) -> np.ndarray:
        ret = np.zeros(self.expr_dim, dtype=np.float32)
        data_type = self.db_info.data_types[node.left.column]
        ret[self.data_type_offset + data_type.value] = 1
        ret[self.left_table_offset + node.left.table] = 1
        ret[self.left_column_offset + node.left.column] = 1
        ret[self.op_offset + int(node.op.positive)] = 1
        ret[self.op_offset + 2 + node.op.op.value] = 1
        if type(node.right) == Column:
            ret[self.right_table_offset + node.right.table] = 1
            ret[self.right_column_offset + node.right.column] = 1
        elif type(node.right) == Number:
            ret[self.number_offset] = node.right.value
            ret[self.array_offset] = 1 / self.max_array_len
        elif type(node.right) == Array:
            ret[self.array_offset] = node.right.size / self.max_array_len
        elif type(node.right) == Null:
            ret[self.null_offset] = 1
        else:
            raise RuntimeError(f'Unknonw Entity: {node.right}')
        return ret