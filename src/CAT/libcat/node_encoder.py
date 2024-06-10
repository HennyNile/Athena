from enum import Enum

import torch

class KeyType(Enum):
    NodeType     = 0
    RelationName = 1
    Alias        = 2
    HashCond     = 3
    IndexCond    = 4
    RecheckCond  = 5
    MergeCond    = 6
    Filter       = 7
    JoinFilter   = 8
    InnerUnique  = 9
    CacheKey     = 10
    SortKey      = 11

class NodeType(Enum):
    Gather          = 0
    Materialize     = 1
    NestedLoop      = 2
    Memoize         = 3
    HashJoin        = 4
    Hash            = 5
    MergeJoin       = 6
    Sort            = 7
    GatherMerge     = 8
    SeqScan         = 9
    IndexScan       = 10
    IndexOnlyScan   = 11
    BitmapHeapScan  = 12
    BitmapIndexScan = 13

def batch_seqs(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    max_seq_len = max(seq.shape[0] for seq in seqs)
    batch = torch.zeros((len(seqs), max_seq_len, seqs[0].shape[1]), dtype=torch.float, device=seqs[0].device)
    lens = []
    for idx, seq in enumerate(seqs):
        batch[idx, :seq.shape[0]] = seq
        lens.append(seq.shape[0])
    return batch, lens

class EmbeddingManager:
    def __init__(self, seqs, model, batch_size: int = 64) -> None:
        self.available_begin = 0
        self.available_end = 0
        self.next = 0
        self.current_embeddings = None
        self.current_embeddings_cpu = None
        self.seqs = seqs
        self.model = model
        self.batch_size = batch_size

    def get_embedding(self, idx: int, device: torch.device) -> torch.Tensor:
        assert idx == self.next, f"Expected idx {self.next} but got {idx}"
        if idx < self.available_begin:
            raise RuntimeError(f"Embedding {idx} is not available")
        elif idx < self.available_end:
            self.next += 1
            ret = self.current_embeddings[idx - self.available_begin]
        else:
            x, seq_lens = batch_seqs(self.seqs[self.available_end:self.available_end + self.batch_size])
            self.current_embeddings = self.model(x.to(device), seq_lens)
            self.available_begin = self.available_end
            self.available_end += self.batch_size
            self.next += 1
            ret = self.current_embeddings[0]
        return ret

def transform_gather(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((1, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.Gather.value] = 1.
    return ret

def transform_materialize(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((1, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.Materialize.value] = 1.
    return ret

def transform_memoize(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((4, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.Memoize.value] = 1.
    ret[1, KeyType.RelationName.value] = 1.
    alias, column_name = node['Cache Key'].split('.')
    relation_name, relation_idx = alias_map[alias]
    ret[1, num_key_types + table_map[relation_name]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + relation_idx] = 1.
    ret[3, KeyType.CacheKey.value] = 1.
    ret[3, num_key_types + column_map[(relation_name, column_name)]] = 1.
    return ret

def transform_nested_loop(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((3, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.NestedLoop.value] = 1.
    ret[1, KeyType.InnerUnique.value] = 1.
    ret[1, num_key_types + 1 if node['Inner Unique'] else 0] = 1.
    ret[2, KeyType.JoinFilter.value] = 1.
    if 'Join Filter' in node:
        ret[2, num_key_types:] = embedding_manager.get_embedding(node['Join Filter'], device)
    return ret

def transform_hash_join(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((3, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.HashJoin.value] = 1.
    ret[1, KeyType.InnerUnique.value] = 1.
    ret[1, num_key_types + 1 if node['Inner Unique'] else 0] = 1.
    ret[2, KeyType.HashCond.value] = 1.
    ret[2, num_key_types:] = embedding_manager.get_embedding(node['Hash Cond'], device)
    return ret

def transform_hash(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((1, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.Hash.value] = 1.
    return ret

def transform_merge_join(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((4, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.MergeJoin.value] = 1.
    ret[1, KeyType.InnerUnique.value] = 1.
    ret[1, num_key_types + 1 if node['Inner Unique'] else 0] = 1.
    ret[2, KeyType.MergeCond.value] = 1.
    ret[2, num_key_types:] = embedding_manager.get_embedding(node['Merge Cond'], device)
    ret[3, KeyType.JoinFilter.value] = 1.
    if 'Join Filter' in node:
        ret[3, num_key_types:] = embedding_manager.get_embedding(node['Join Filter'], device)
    return ret

def transform_sort(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((4, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.Sort.value] = 1.
    assert len(node['Sort Key']) >= 1
    ret[1, KeyType.RelationName.value] = 1.
    alias, column_name = node['Sort Key'][0].split('.')
    relation_name, relation_idx = alias_map[alias]
    ret[1, num_key_types + table_map[relation_name]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + relation_idx] = 1.
    ret[3, KeyType.SortKey.value] = 1.
    ret[3, num_key_types + column_map[(relation_name, column_name)]] = 1.
    return ret

def transform_gather_merge(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((1, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.GatherMerge.value] = 1.
    return ret

def transform_seq_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((4, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.SeqScan.value] = 1.
    ret[1, KeyType.RelationName.value] = 1.
    ret[1, num_key_types + table_map[node['Relation Name']]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + alias_map[node['Alias']][1]] = 1.
    ret[3, KeyType.Filter.value] = 1.
    if 'Filter' in node:
        ret[3, num_key_types:] = embedding_manager.get_embedding(node['Filter'], device)
    return ret

def transform_index_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((5, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.IndexScan.value] = 1.
    ret[1, KeyType.RelationName.value] = 1.
    ret[1, num_key_types + table_map[node['Relation Name']]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + alias_map[node['Alias']][1]] = 1.
    ret[3, KeyType.IndexCond.value] = 1.
    if 'Index Cond' in node:
        ret[3, num_key_types:] = embedding_manager.get_embedding(node['Index Cond'], device)
    ret[4, KeyType.Filter.value] = 1.
    if 'Filter' in node:
        ret[4, num_key_types:] = embedding_manager.get_embedding(node['Filter'], device)
    return ret

def transform_index_only_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((4, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.IndexOnlyScan.value] = 1.
    ret[1, KeyType.RelationName.value] = 1.
    ret[1, num_key_types + table_map[node['Relation Name']]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + alias_map[node['Alias']][1]] = 1.
    ret[3, KeyType.IndexCond.value] = 1.
    if 'Index Cond' in node:
        ret[3, num_key_types:] = embedding_manager.get_embedding(node['Index Cond'], device)
    return ret

def transform_bitmap_heap_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((5, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.BitmapHeapScan.value] = 1.
    ret[1, KeyType.RelationName.value] = 1.
    ret[1, num_key_types + table_map[node['Relation Name']]] = 1.
    ret[2, KeyType.Alias.value] = 1.
    ret[2, num_key_types + alias_map[node['Alias']][1]] = 1.
    ret[3, KeyType.RecheckCond.value] = 1.
    if 'Recheck Cond' in node:
        ret[3, num_key_types:] = embedding_manager.get_embedding(node['Recheck Cond'], device)
    ret[4, KeyType.Filter.value] = 1.
    if 'Filter' in node:
        ret[4, num_key_types:] = embedding_manager.get_embedding(node['Filter'], device)
    return ret

def transform_bitmap_index_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager: 'EmbeddingManager', device: torch.device):
    num_key_types = len(KeyType)
    ret = torch.zeros((2, num_key_types + expr_embedding_dim), dtype=torch.float, device=device)
    ret[0, KeyType.NodeType.value] = 1.
    ret[0, num_key_types + NodeType.BitmapIndexScan.value] = 1.
    ret[1, KeyType.IndexCond.value] = 1.
    ret[1, num_key_types:] = embedding_manager.get_embedding(node['Index Cond'], device)
    return ret

def featurize_nodes(sample, expr_embedding_dim: int, alias_map, table_map, column_map, embedding_manager, device) -> tuple[list[torch.Tensor], list[list[int]], list[list[int]], list[int]]:
    root = sample['Plan']

    while 'Plans' in root and len(root['Plans']) == 1:
        root = root['Plans'][0]

    level_indices: list[int] = []
    attr_sets = []
    pos_encodings = []
    masks = []
    cards = []
    costs = []
    def dfs(node, parent=None, level=0, leaves_begin=0, nodes_begin=0) -> tuple[int, int]:
        if level == len(level_indices):
            level_indices.append(0)
        else:
            level_indices[level] += 1
        node_type = node['Node Type']
        match node_type:
            case 'Gather':
                attr_sets.append(transform_gather(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Materialize':
                attr_sets.append(transform_materialize(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Nested Loop':
                attr_sets.append(transform_nested_loop(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Memoize':
                attr_sets.append(transform_memoize(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Hash Join':
                attr_sets.append(transform_hash_join(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Hash':
                attr_sets.append(transform_hash(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Merge Join':
                attr_sets.append(transform_merge_join(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Sort':
                attr_sets.append(transform_sort(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Gather Merge':
                attr_sets.append(transform_gather_merge(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Seq Scan':
                attr_sets.append(transform_seq_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Index Scan':
                attr_sets.append(transform_index_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Index Only Scan':
                attr_sets.append(transform_index_only_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Bitmap Heap Scan':
                attr_sets.append(transform_bitmap_heap_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case 'Bitmap Index Scan':
                attr_sets.append(transform_bitmap_index_scan(node, expr_embedding_dim, alias_map, table_map, column_map, embedding_manager, device))
            case _:
                raise NotImplementedError

        pos_encodings.append([])
        masks.append([])
        if 'Execution Time' in sample:
            cards.append(node['Actual Rows'] * node['Actual Loops'])
        else:
            cards.append(-1.)
        pos_encoding = pos_encodings[-1]
        mask = masks[-1]
        pos_encoding.append(level)
        pos_encoding.append(level_indices[level])
        pos_encoding.append(leaves_begin)
        mask.append(nodes_begin)
        nodes_begin += 1
        if 'Plans' in node:
            for child in node['Plans']:
                leaves_begin, nodes_begin = dfs(child, node, level + 1, leaves_begin, nodes_begin)
        else:
            leaves_begin += 1
        pos_encoding.append(leaves_begin)
        mask.append(nodes_begin)
        return leaves_begin, nodes_begin

    dfs(root)
    return attr_sets, pos_encodings, masks, cards
