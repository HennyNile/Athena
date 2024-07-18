from datetime import datetime
from enum import Enum
import math
import sys

import torch
from tqdm import tqdm

from model import CatModel

sys.path.append('.')
from src.utils import tokenizer as plan_expr_parser
from src.utils.tokenizer import TokenType
from src.utils.db_utils import DBInfo

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
    PlanRows     = 12

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

class SyntaxType(Enum):
    LEFT_PARAN      = 0
    RIGHT_PARAN     = 1
    LEFT_BRACKET    = 2
    RIGHT_BRACKET   = 3
    ARRAY_BEGIN     = 4
    ARRAY_END       = 5
    EQUAL           = 6
    LESS_THAN       = 7
    GREATER_THAN    = 8
    LESS_EQUAL      = 9
    GREATER_EQUAL   = 10
    NOT_EQUAL       = 11
    DOT             = 12
    COMMA           = 13
    SINGLE_QUOTE    = 14
    DOUBLE_QUOTE    = 15
    WILDCARD        = 16
    ATTRIBUTE       = 17
    OP_LIKE         = 18
    OP_NOT_LIKE     = 19
    KEYWORD_TEXT    = 20
    KEYWORD_ANY     = 21
    KEYWORD_AND     = 22
    KEYWORD_OR      = 23
    KEYWORD_IS      = 24
    KEYWORD_NOT     = 25
    KEYWORD_NULL    = 26
    FUNCTION_SUBSTR = 27
    KEYWORD_INTEGER = 28

keys = [
    "Node Type",
    "Relation Name",
    "Alias",
    "Hash Cond",
    "Index Cond",
    "Recheck Cond",
    "Merge Cond",
    "Filter",
    "Join Filter",
    "Inner Unique",
    "Cache Key",
    "Sort Key",
]

node_types = [
    "Gather",
    "Materialize",
    "Nested Loop",
    "Memoize",
    "Hash Join",
    "Hash",
    "Merge Join",
    "Sort",
    "Gather Merge",
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
]

syntax_types = [
    "LEFT_PARAN",
    "RIGHT_PARAN",
    "LEFT_BRACKET",
    "RIGHT_BRACKET",
    "ARRAY_BEGIN",
    "ARRAY_END",
    "EQUAL",
    "LESS_THAN",
    "GREATER_THAN",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "NOT_EQUAL",
    "DOT",
    "COMMA",
    "SINGLE_QUOTE",
    "DOUBLE_QUOTE",
    "WILDCARD",
    "ATTRIBUTE",
    "OP_LIKE",
    "OP_NOT_LIKE",
    "KEYWORD_TEXT",
    "KEYWORD_ANY",
    "KEYWORD_AND",
    "KEYWORD_OR",
    "KEYWORD_IS",
    "KEYWORD_NOT",
    "KEYWORD_NULL",
]

def find_id(vec: list[str], value: str) -> int:
    for idx, v in enumerate(vec):
        if v == value:
            return idx
    raise ValueError(f'{value} not found in {vec}')

def key_to_id(key: str) -> KeyType:
    return KeyType(find_id(keys, key))

def node_type_to_id(node_type: str) -> NodeType:
    return NodeType(find_id(node_types, node_type))

def key_to_string(key: KeyType) -> str:
    return keys[key.value]

def node_type_to_string(node_type: NodeType) -> str:
    return node_types[node_type.value]

def syntax_type_to_string(syntax_type: SyntaxType) -> str:
    return syntax_types[syntax_type.value]

class CLSToken:
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return 'CLS'

class KeyToken:
    def __init__(self, key: KeyType) -> None:
        self.key = key

    def __str__(self) -> str:
        return f'Key{{{key_to_string(self.key)}}}'

class NodeToken:
    def __init__(self, node_type: NodeType) -> None:
        self.node_type = node_type

    def __str__(self) -> str:
        return f'Node{{{node_type_to_string(self.node_type)}}}'

class RelationToken:
    def __init__(self, relation_id: int) -> None:
        self.relation_id = relation_id

    def __str__(self) -> str:
        return f'Relation{{{self.relation_id}}}'

class AliasToken:
    def __init__(self, alias_id: int) -> None:
        self.alias_id = alias_id
    
    def __str__(self) -> str:
        return f'Alias{{{self.alias_id}}}'

class ColumnToken:
    def __init__(self, column_id: int) -> None:
        self.column_id = column_id

    def __str__(self) -> str:
        return f'Column{{{self.column_id}}}'

class InnerUniqueToken:
    def __init__(self, inner_unique: bool) -> None:
        self.inner_unique = inner_unique

    def __str__(self) -> str:
        return f'InnerUnique{{{self.inner_unique}}}'

class WordToken:
    def __init__(self, word: str) -> None:
        self.word = word

    def __str__(self) -> str:
        return f'Word{{{self.word}}}'

class SyntaxToken:
    def __init__(self, syntax: SyntaxType) -> None:
        self.syntax = syntax

    def __str__(self) -> str:
        return f'Syntax{{{syntax_type_to_string(self.syntax)}}}'

class NumberToken:
    def __init__(self, number: float) -> None:
        self.number = number

    def __str__(self) -> str:
        return f'Number{{{self.number}}}'

class TimestampToken:
    def __init__(self, s: str) -> None:
        try:
            self.timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.timestamp = datetime.strptime(s, '%Y-%m-%d')

    def __str__(self) -> str:
        return f'Timestamp{{{self.timestamp}}}'

class CardinalityToken:
    def __init__(self, card: float) -> None:
        self.card = card

    def __str__(self) -> str:
        return f'Cardinality{{{self.card}}}'

def word_splitter(word):
    words = plan_expr_parser.string_splitter(word)
    ret = []
    for wt, word in words:
        match wt:
            case TokenType.WORD:
                ret.append(WordToken(word))
            case TokenType.LEFT_PARAN:
                ret.append(SyntaxToken(SyntaxType.LEFT_PARAN))
            case TokenType.RIGHT_PARAN:
                ret.append(SyntaxToken(SyntaxType.RIGHT_PARAN))
            case TokenType.LEFT_BRACKET:
                ret.append(WordToken('['))
            case TokenType.RIGHT_BRACKET:
                ret.append(WordToken(']'))
            case TokenType.WILDCARD:
                ret.append(SyntaxToken(SyntaxType.WILDCARD))
            case TokenType.COMMA:
                ret.append(WordToken(','))
            case TokenType.HYPHEN:
                ret.append(WordToken('-'))
            case TokenType.UNDER_SCORE:
                ret.append(WordToken('_'))
            case TokenType.COLON:
                ret.append(WordToken(':'))
            case TokenType.ADD:
                ret.append(WordToken('+'))
            case TokenType.HASH:
                ret.append(WordToken('#'))
            case TokenType.GREATER_THAN:
                ret.append(WordToken('>'))
            case _:
                raise NotImplementedError
    return ret

class PlanInfo:
    def __init__(self, alias_map: dict[str, tuple[str, int]], rel_names: set[str]) -> None:
        self.alias_map = alias_map
        self.rel_names = rel_names

def tokenizer(expr: str, plan_info: PlanInfo, db_info: DBInfo, node: dict, node_parent: dict|None) -> tuple[list, int]:
    skip = 0
    max_alias_idx = 0
    tokens = plan_expr_parser.tokenizer(expr)
    ret = []
    for idx, (t, token) in enumerate(tokens):
        if skip > 0:
            skip -= 1
            continue
        match t:
            case TokenType.LEFT_PARAN:
                ret.append(SyntaxToken(SyntaxType.LEFT_PARAN))
            case TokenType.RIGHT_PARAN:
                ret.append(SyntaxToken(SyntaxType.RIGHT_PARAN))
            case TokenType.LEFT_BRACKET:
                ret.append(SyntaxToken(SyntaxType.LEFT_BRACKET))
            case TokenType.RIGHT_BRACKET:
                ret.append(SyntaxToken(SyntaxType.RIGHT_BRACKET))
            case TokenType.ARRAY_BEGIN:
                ret.append(SyntaxToken(SyntaxType.ARRAY_BEGIN))
            case TokenType.ARRAY_END:
                ret.append(SyntaxToken(SyntaxType.ARRAY_END))
            case TokenType.EQUAL:
                ret.append(SyntaxToken(SyntaxType.EQUAL))
            case TokenType.LESS_THAN:
                ret.append(SyntaxToken(SyntaxType.LESS_THAN))
            case TokenType.GREATER_THAN:
                ret.append(SyntaxToken(SyntaxType.GREATER_THAN))
            case TokenType.LESS_EQUAL:
                ret.append(SyntaxToken(SyntaxType.LESS_EQUAL))
            case TokenType.GREATER_EQUAL:
                ret.append(SyntaxToken(SyntaxType.GREATER_EQUAL))
            case TokenType.NOT_EQUAL:
                ret.append(SyntaxToken(SyntaxType.NOT_EQUAL))
            case TokenType.DOT:
                ret.append(SyntaxToken(SyntaxType.DOT))
            case TokenType.COMMA:
                ret.append(SyntaxToken(SyntaxType.COMMA))
            case TokenType.ATTRIBUTE:
                ret.append(SyntaxToken(SyntaxType.ATTRIBUTE))
            case TokenType.LIKE:
                ret.append(SyntaxToken(SyntaxType.OP_LIKE))
            case TokenType.NOT_LIKE:
                ret.append(SyntaxToken(SyntaxType.OP_NOT_LIKE))
            case TokenType.SINGLE_QUOTE:
                if idx + 2 < len(tokens) and tokens[idx + 1][0] == TokenType.ATTRIBUTE and tokens[idx + 2][0] == TokenType.IDENTIFIER:
                    if tokens[idx + 2][1] == "integer":
                        if len(ret) >= 2 and type(ret[-2]) == ColumnToken:
                            m, M = db_info.get_normalizer(ret[-2].column_id)
                            value = float(token[1:-1])
                            value = (value - m) / (M - m)
                            ret.append(NumberToken(value))
                        else:
                            raise RuntimeError("Number should be a column value")
                        skip = 2
                    elif tokens[idx + 2][1] == "numeric":
                        if len(ret) >= 2 and type(ret[-2]) == ColumnToken:
                            m, M = db_info.get_normalizer(ret[-2].column_id)
                            value = float(token[1:-1])
                            value = (value - float(m)) / float(M - m)
                            ret.append(NumberToken(value))
                        else:
                            raise RuntimeError("Number should be a column value")
                        skip = 2
                    elif tokens[idx + 2][1] == "date":
                        ret.append(TimestampToken(token[1:-1]))
                        skip = 2
                    elif tokens[idx + 2][1] == "timestamp":
                        ret.append(TimestampToken(token[1:-1]))
                        skip = 5
                    elif tokens[idx + 2][1] == "text":
                        ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
                        ret.extend(word_splitter(token[1:-1]))
                        ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
                    elif tokens[idx + 2][1] == "bpchar":
                        ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
                        ret.extend(word_splitter(token[1:-1]))
                        ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
                    else:
                        raise RuntimeError(f"Unknown attribute: {tokens[idx + 2][1]} for token {token}")
                else:
                    ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
                    ret.extend(word_splitter(token[1:-1]))
                    ret.append(SyntaxToken(SyntaxType.SINGLE_QUOTE))
            case TokenType.DOUBLE_QUOTE:
                ret.append(SyntaxToken(SyntaxType.DOUBLE_QUOTE))
                ret.extend(word_splitter(token[1:-1]))
                ret.append(SyntaxToken(SyntaxType.DOUBLE_QUOTE))
            case TokenType.IDENTIFIER:
                if token in plan_info.alias_map:
                    table_name, alias_idx = plan_info.alias_map[token]
                    ret.append(RelationToken(db_info.table_map[table_name]))
                    ret.append(AliasToken(alias_idx))
                    max_alias_idx = max(max_alias_idx, alias_idx)
                elif token in plan_info.rel_names:
                    ret.append(RelationToken(db_info.table_map[token]))
                    ret.append(AliasToken(0))
                elif token == 'substr':
                    ret.append(SyntaxToken(SyntaxType.FUNCTION_SUBSTR))
                elif token == 'ANY':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_ANY))
                elif idx > 0 and (tokens[idx - 1][0] == TokenType.DOT or tokens[idx - 1][0] == TokenType.EQUAL or tokens[idx - 1][0] == TokenType.LEFT_PARAN):
                    if len(ret) >= 3 and type(ret[-1]) == SyntaxToken and ret[-1].syntax == SyntaxType.DOT and type(ret[-3]) == RelationToken:
                        relation_id = ret[-3].relation_id
                    elif 'Relation Name' in node:
                        relation_id = db_info.table_map[node['Relation Name']]
                    elif node_parent is not None and 'Relation Name' in node_parent:
                        relation_id = db_info.table_map[node_parent['Relation Name']]
                    else:
                        raise RuntimeError(f"Unknown column name: {token} in {expr}")
                    try:
                        column_id = db_info.column_map_list[relation_id][token]
                    except Exception as e:
                        print(db_info.column_map_list[relation_id])
                        raise e
                    ret.append(ColumnToken(column_id))
                elif token == 'text':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_TEXT))
                elif token == 'bpchar':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_TEXT))
                elif token == 'integer':
                    begin_idx = -1
                    while not (type(ret[begin_idx]) == SyntaxToken and ret[begin_idx].syntax == SyntaxType.ARRAY_BEGIN):
                        begin_idx -= 1
                    new_ret = ret[:begin_idx + 1]
                    print(db_info.column_map)
                    print(ret[begin_idx - 4])
                    m, M = db_info.get_normalizer(ret[begin_idx - 4].column_id)
                    print(m, M)
                    current_word = ''
                    for array_idx in range(begin_idx + 1, -2):
                        if type(ret[array_idx]) == WordToken:
                            current_word += ret[array_idx].word
                        elif type(ret[array_idx]) == SyntaxToken and (ret[array_idx].syntax == SyntaxType.COMMA or ret[array_idx].syntax == SyntaxType.ARRAY_END):
                            value = float(current_word)
                            value = (value - float(m)) / float(M - m)
                            new_ret.append(NumberToken(value))
                            new_ret.append(ret[array_idx])
                            current_word = ''
                        else:
                            raise RuntimeError(f'Unexpected token type')
                    ret = new_ret
                    ret.append(SyntaxToken(SyntaxType.ATTRIBUTE))
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_INTEGER))
                elif token == 'AND':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_AND))
                elif token == 'OR':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_OR))
                elif token == 'IS':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_IS))
                elif token == 'NOT':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_NOT))
                elif token == 'NULL':
                    ret.append(SyntaxToken(SyntaxType.KEYWORD_NULL))
                else:
                    raise NotImplementedError(f'{token}, {tokens[idx - 1]}')
            case TokenType.NUMBER:
                if len(ret) >= 2 and type(ret[-2]) == ColumnToken:
                    m, M = db_info.get_normalizer(ret[-2].column_id)
                    value = float(token)
                    value = (value - float(m)) / float(M - m)
                    ret.append(NumberToken(value))
                else:
                    pass
            case TokenType.WORD:
                ret.extend(word_splitter(token))
            case _:
                raise NotImplementedError
    return ret, max_alias_idx

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

class Node:
    def __init__(self, token_begin: int, token_end: int, depth: int, node_begin: int, node_end: int, leaf_begin: int, leaf_end: int, card: float) -> None:
        self.token_begin = token_begin
        self.token_end = token_end
        self.depth = depth
        self.node_begin = node_begin
        self.node_end = node_end
        self.leaf_begin = leaf_begin
        self.leaf_end = leaf_end
        self.card = card

class Sample:
    def __init__(self, tokens: list[tuple[int, float]], nodes: list[Node], cost: float, exe_time: float, weight: float = 1.) -> None:
        self.tokens = tokens
        self.nodes = nodes
        self.cost = cost
        self.exe_time = exe_time
        self.weight = weight

class Input:
    def __init__(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, node_pos: torch.Tensor, node_mask: torch.Tensor, output_idx: torch.Tensor, cards: torch.Tensor, cost: float|torch.Tensor, exe_time: float|torch.Tensor, weight: float|torch.Tensor) -> None:
        self.x = x
        self.pos = pos
        self.mask = mask
        self.node_pos = node_pos
        self.node_mask = node_mask
        self.output_idx = output_idx
        self.cards = cards
        self.cost = cost
        self.exe_time = exe_time
        self.weight = weight

    def cuda(self) -> 'Input':
        x_cuda = self.x.cuda()
        pos_cuda = self.pos.cuda()
        mask_cuda = self.mask.cuda()
        node_pos_cuda = self.node_pos.cuda()
        node_mask_cuda = self.node_mask.cuda()
        output_idx_cuda = self.output_idx.cuda()
        cards_cuda = self.cards.cuda()
        if type(self.cost) == torch.Tensor:
            cost_cuda = self.cost.cuda()
        else:
            cost_cuda = self.cost
        if type(self.exe_time) == torch.Tensor:
            exe_time_cuda = self.exe_time.cuda()
        else:
            exe_time_cuda = self.exe_time
        if type(self.weight) == torch.Tensor:
            weight_cuda = self.weight.cuda()
        else:
            weight_cuda = self.weight
        return Input(x_cuda, pos_cuda, mask_cuda, node_pos_cuda, node_mask_cuda, output_idx_cuda, cards_cuda, cost_cuda, exe_time_cuda, weight_cuda)
    
    def save(self, path: str) -> None:
        obj = {
            "x": self.x,
            "pos": self.pos,
            "mask": self.mask,
            "node_pos": self.node_pos,
            "node_mask": self.node_mask,
            "output_idx": self.output_idx,
            "cards": self.cards,
            "cost": self.cost,
            "exe_time": self.exe_time,
            "weight": self.weight
        }
        torch.save(obj, path)

    @staticmethod
    def load(path: str) -> 'Input':
        obj = torch.load(path)
        return Input(obj["x"], obj["pos"], obj["mask"], obj["node_pos"], obj["node_mask"], obj["output_idx"], obj["cards"], obj["cost"], obj["exe_time"], obj["weight"])

class Cat:
    def __init__(self, db_info: DBInfo):
        self.db_info = db_info
        self.word_table: dict[str, int] = {}
        self.max_alias_idx: int = 0
        self.max_card: float = 0.
        self.model: torch.nn.Module = None

    def fit_all(self, dataset: list[list[dict]]) -> None:
        # word_table and max_alias_idx
        plans = [sample for query in dataset for sample in query]
        for sample in plans:
            plan_info, _ = get_alias_map(sample)
            self._fit_sample(sample, plan_info)
        self.cls_offset = 0
        self.number_offset       = self.cls_offset          + 1
        self.key_offset          = self.number_offset       + 1
        self.node_offset         = self.key_offset          + len(KeyType)
        self.relation_offset     = self.node_offset         + len(NodeType)
        self.alias_offset        = self.relation_offset     + len(self.db_info.table_map)
        self.column_offset       = self.alias_offset        + self.max_alias_idx + 1
        self.inner_unique_offset = self.column_offset       + len(self.db_info.column_map)
        self.word_offset         = self.inner_unique_offset + 2
        self.syntax_offset       = self.word_offset         + len(self.word_table)
        self.card_offset         = self.syntax_offset       + len(SyntaxType)
        self.feature_dim         = self.card_offset         + 1
        print(f'cls_offset: {self.cls_offset}')
        print(f'number_offset: {self.number_offset}')
        print(f'key_offset: {self.key_offset}')
        print(f'node_offset: {self.node_offset}')
        print(f'relation_offset: {self.relation_offset}')
        print(f'alias_offset: {self.alias_offset}')
        print(f'column_offset: {self.column_offset}')
        print(f'inner_unique_offset: {self.inner_unique_offset}')
        print(f'word_offset: {self.word_offset}')
        print(f'syntax_offset: {self.syntax_offset}')
        print(f'card_offset: {self.card_offset}')
        print(f'feature_dim: {self.feature_dim}')

    def fit_train(self, dataset: list[list[dict]]) -> None:
        # max_card
        plans = [sample for query in dataset for sample in query]
        for sample in plans:
            _, max_card = get_alias_map(sample)
            self.max_card = max(self.max_card, float(max_card))
        print(f'max_card: {self.max_card}')

    def init_model(self) -> None:
        self.model = CatModel(self.feature_dim, 128, 8, 256, 6)

    def transform(self, plans: list[dict], num_finished: list[int] = None) -> list[Sample]:
        if num_finished is None:
            num_finished = [1 for _ in plans]
        assert len(num_finished) == len(plans)
        print("Preparing samples:")
        return [self._featurize_plan(plan, 1. / num) for plan, num in zip(tqdm(plans), num_finished)]

    def transform_sample(self, sample: Sample) -> Input:
        len_token = len(sample.tokens)
        len_node = len(sample.nodes)
        x = torch.zeros(len_token, self.feature_dim, dtype=torch.float32)
        pos = torch.zeros(len_token, 4, dtype=torch.float32)
        mask = torch.zeros(len_token, len_token, dtype=torch.float32)
        node_pos = torch.zeros(len_node, 4, dtype=torch.float32)
        node_mask = torch.zeros(len_node, len_node, dtype=torch.float32)
        output_idx = torch.zeros(len_node, dtype=torch.long)
        cards = torch.log(torch.tensor([node.card for node in sample.nodes], dtype=torch.float32) + 1.) / math.log(self.max_card + 1.)
        cost = sample.cost
        weight = sample.weight
        for node_idx, node in enumerate(sample.nodes):
            for token_idx in range(node.token_begin, node.token_end):
                token = sample.tokens[token_idx]
                match token:
                    case CLSToken():
                        x[token_idx, self.cls_offset] = 1.
                    case KeyToken(key=key):
                        x[token_idx, self.key_offset + key.value] = 1.
                    case NodeToken(node_type=node_type):
                        x[token_idx, self.node_offset + node_type.value] = 1.
                    case RelationToken(relation_id=relation_id):
                        x[token_idx, self.relation_offset + relation_id] = 1.
                    case AliasToken(alias_id=alias_id):
                        x[token_idx, self.alias_offset + alias_id] = 1.
                    case ColumnToken(column_id=column_id):
                        x[token_idx, self.column_offset + column_id] = 1.
                    case InnerUniqueToken(inner_unique=inner_unique):
                        x[token_idx, self.inner_unique_offset + int(inner_unique)] = 1.
                    case WordToken(word=word):
                        x[token_idx, self.word_offset + self.word_table[word]] = 1.
                    case SyntaxToken(syntax=syntax):
                        x[token_idx, self.syntax_offset + syntax.value] = 1.
                    case NumberToken(number=number):
                        x[token_idx, self.number_offset] = number
                    case TimestampToken(timestamp=timestamp):
                        x[token_idx, self.number_offset] = self._norm_timestamp(timestamp)
                    case CardinalityToken(card=card):
                        x[token_idx, self.card_offset] = card
                    case _:
                        print(token)
                        raise NotImplementedError
                pos[token_idx] = torch.tensor([node.depth, node.leaf_begin, node.leaf_end, token_idx - node.token_begin], dtype=torch.float32)
                mask[token_idx].fill_(-torch.inf)
                mask[token_idx, node.token_begin:sample.nodes[node.node_end - 1].token_end] = 0.
            node_pos[node_idx] = torch.tensor([node.depth, node.leaf_begin, node.leaf_end, 0], dtype=torch.float32)
            node_mask[node_idx].fill_(-torch.inf)
            node_mask[node_idx, node.node_begin:node.node_end] = 0.
            output_idx[node_idx] = node.token_begin
        return Input(x, pos, mask, node_pos, node_mask, output_idx, cards, cost, sample.exe_time, weight)

    def batch_transformed_samples(self, inputs: list[Input]):
        batch_size = len(inputs)
        max_token_len = max(input.x.shape[0] for input in inputs)
        max_node_len = max(input.node_pos.shape[0] for input in inputs)
        ret_x = torch.zeros(batch_size, max_token_len, self.feature_dim, dtype=torch.float32)
        ret_pos = torch.zeros(batch_size, max_token_len, 4, dtype=torch.float32)
        ret_mask = torch.zeros(batch_size, max_token_len, max_token_len, dtype=torch.float32)
        ret_node_pos = torch.zeros(batch_size, max_node_len, 4, dtype=torch.float32)
        ret_node_mask = torch.zeros(batch_size, max_node_len, max_node_len, dtype=torch.float32)
        ret_output_idx = torch.zeros(batch_size, max_node_len, dtype=torch.long)
        ret_cards = torch.zeros(batch_size, max_node_len, dtype=torch.float32)
        ret_cost = torch.tensor([input.cost for input in inputs], dtype=torch.float32)
        ret_exe_time = torch.tensor([input.exe_time for input in inputs], dtype=torch.float32)
        ret_weight = torch.tensor([input.weight for input in inputs], dtype=torch.float32)

        for idx, input in enumerate(inputs):
            ret_x[idx, :input.x.shape[0]] = input.x
            ret_pos[idx, :input.pos.shape[0]] = input.pos
            ret_mask[idx, :input.mask.shape[0], :input.mask.shape[1]] = input.mask
            ret_mask[idx, :input.mask.shape[0], input.mask.shape[1]:] = -torch.inf
            ret_node_pos[idx, :input.node_pos.shape[0]] = input.node_pos
            ret_node_mask[idx, :input.node_mask.shape[0], :input.node_mask.shape[1]] = input.node_mask
            ret_node_mask[idx, :input.node_mask.shape[0], input.node_mask.shape[1]:] = -torch.inf
            ret_output_idx[idx, :input.output_idx.shape[0]] = input.output_idx
            ret_cards[idx, :input.cards.shape[0]] = input.cards

        return Input(ret_x, ret_pos, ret_mask, ret_node_pos, ret_node_mask, ret_output_idx, ret_cards, ret_cost, ret_exe_time, ret_weight)

    def transform_samples(self, batch: list[Sample]):
        inputs = [self.transform_sample(sample) for sample in batch]
        return self.batch_transformed_samples(inputs)

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
            'max_alias_idx': self.max_alias_idx,
            'max_card': self.max_card,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        model_state = torch.load(path)
        self.word_table = model_state['word_table']
        self.max_alias_idx = model_state['max_alias_idx']
        self.max_card = model_state['max_card']
        self.cls_offset = 0
        self.number_offset       = self.cls_offset          + 1
        self.key_offset          = self.number_offset       + 1
        self.node_offset         = self.key_offset          + len(KeyType)
        self.relation_offset     = self.node_offset         + len(NodeType)
        self.alias_offset        = self.relation_offset     + len(self.db_info.table_map)
        self.column_offset       = self.alias_offset        + self.max_alias_idx + 1
        self.inner_unique_offset = self.column_offset       + len(self.db_info.column_map)
        self.word_offset         = self.inner_unique_offset + 2
        self.syntax_offset       = self.word_offset         + len(self.word_table)
        self.card_offset         = self.syntax_offset       + len(SyntaxType)
        self.feature_dim         = self.card_offset         + 1
        self.init_model()
        self.model.load_state_dict(model_state['state_dict'])

    def _fit_sample(self, sample: dict, plan_info: PlanInfo) -> None:
        root = sample['Plan']
        seqs = []

        def dfs(node, parent=None):
            if 'Hash Cond' in node:
                seqs.append(tokenizer(node['Hash Cond'], plan_info, self.db_info, node, parent))
            if 'Index Cond' in node:
                seqs.append(tokenizer(node['Index Cond'], plan_info, self.db_info, node, parent))
            if 'Recheck Cond' in node:
                seqs.append(tokenizer(node['Recheck Cond'], plan_info, self.db_info, node, parent))
            if 'Merge Cond' in node:
                seqs.append(tokenizer(node['Merge Cond'], plan_info, self.db_info, node, parent))
            if 'Filter' in node:
                seqs.append(tokenizer(node['Filter'], plan_info, self.db_info, node, parent))
            if 'Join Filter' in node:
                seqs.append(tokenizer(node['Join Filter'], plan_info, self.db_info, node, parent))
            for child in node.get('Plans', []):
                dfs(child, node)

        dfs(root)

        for tokens, max_alias_idx in seqs:
            self.max_alias_idx = max(self.max_alias_idx, max_alias_idx)
            for token in tokens:
                if type(token) == WordToken:
                    if token.word not in self.word_table:
                        self.word_table[token.word] = len(self.word_table)

    def _norm_timestamp(self, dt: datetime) -> float:
        min_t = self.db_info.min_timestamp.timestamp()
        max_t = self.db_info.max_timestamp.timestamp()
        return (dt.timestamp() - min_t) / (max_t - min_t)

    def _featurize_plan(self, plan: dict, weight: float = 1.) -> Sample:
        plan_info, _ = get_alias_map(plan)
        tokens: list = []
        nodes: list[Node] = []

        def dfs(node: dict, parent: dict|None = None, depth: int = 0, node_begin: int = 0, leaf_begin: int = 0) -> int:
            token_begin = len(tokens)
            tokens.append(CLSToken())
            node_type = node_type_to_id(node['Node Type'])
            tokens.append(KeyToken(KeyType.NodeType))
            tokens.append(NodeToken(node_type))
            match node_type:
                case NodeType.Gather:
                    pass
                case NodeType.Materialize:
                    pass
                case NodeType.Hash:
                    pass
                case NodeType.GatherMerge:
                    pass
                case NodeType.Memoize:
                    alias, column = node['Cache Key'].split('.')
                    relation_name, alias_idx = plan_info.alias_map[alias]
                    tokens.append(KeyToken(KeyType.CacheKey))
                    tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                    tokens.append(AliasToken(alias_idx))
                    tokens.append(ColumnToken(self.db_info.column_map[(relation_name, column)]))
                case NodeType.NestedLoop:
                    tokens.append(KeyToken(KeyType.InnerUnique))
                    tokens.append(InnerUniqueToken(node['Inner Unique']))
                    if 'Join Filter' in node:
                        tokens.append(KeyToken(KeyType.JoinFilter))
                        tokens.extend(tokenizer(node['Join Filter'], plan_info, self.db_info, node, parent)[0])
                case NodeType.HashJoin:
                    tokens.append(KeyToken(KeyType.InnerUnique))
                    tokens.append(InnerUniqueToken(node['Inner Unique']))
                    tokens.append(KeyToken(KeyType.HashCond))
                    tokens.extend(tokenizer(node['Hash Cond'], plan_info, self.db_info, node, parent)[0])
                case NodeType.MergeJoin:
                    tokens.append(KeyToken(KeyType.InnerUnique))
                    tokens.append(InnerUniqueToken(node['Inner Unique']))
                    tokens.append(KeyToken(KeyType.MergeCond))
                    tokens.extend(tokenizer(node['Merge Cond'], plan_info, self.db_info, node, parent)[0])
                    if 'Join Filter' in node:
                        tokens.append(KeyToken(KeyType.JoinFilter))
                        tokens.extend(tokenizer(node['Join Filter'], plan_info, self.db_info, node, parent)[0])
                case NodeType.Sort:
                    for key in node['Sort Key']:
                        alias, column = key.split('.')
                        relation_name, alias_idx = plan_info.alias_map[alias]
                        tokens.append(KeyToken(KeyType.SortKey))
                        tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                        tokens.append(AliasToken(alias_idx))
                        tokens.append(ColumnToken(self.db_info.column_map[(relation_name, column)]))
                case NodeType.SeqScan:
                    relation_name, alias_idx = plan_info.alias_map[node['Alias']]
                    tokens.append(KeyToken(KeyType.RelationName))
                    tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                    tokens.append(KeyToken(KeyType.Alias))
                    tokens.append(AliasToken(alias_idx))
                    if 'Filter' in node:
                        tokens.append(KeyToken(KeyType.Filter))
                        tokens.extend(tokenizer(node['Filter'], plan_info, self.db_info, node, parent)[0])
                case NodeType.IndexScan:
                    relation_name, alias_idx = plan_info.alias_map[node['Alias']]
                    tokens.append(KeyToken(KeyType.RelationName))
                    tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                    tokens.append(KeyToken(KeyType.Alias))
                    tokens.append(AliasToken(alias_idx))
                    if 'Index Cond' in node:
                        tokens.append(KeyToken(KeyType.IndexCond))
                        tokens.extend(tokenizer(node['Index Cond'], plan_info, self.db_info, node, parent)[0])
                    if 'Filter' in node:
                        tokens.append(KeyToken(KeyType.Filter))
                        tokens.extend(tokenizer(node['Filter'], plan_info, self.db_info, node, parent)[0])
                case NodeType.IndexOnlyScan:
                    relation_name, alias_idx = plan_info.alias_map[node['Alias']]
                    tokens.append(KeyToken(KeyType.RelationName))
                    tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                    tokens.append(KeyToken(KeyType.Alias))
                    tokens.append(AliasToken(alias_idx))
                    if 'Index Cond' in node:
                        tokens.append(KeyToken(KeyType.IndexCond))
                        tokens.extend(tokenizer(node['Index Cond'], plan_info, self.db_info, node, parent)[0])
                case NodeType.BitmapHeapScan:
                    relation_name, alias_idx = plan_info.alias_map[node['Alias']]
                    tokens.append(KeyToken(KeyType.RelationName))
                    tokens.append(RelationToken(self.db_info.table_map[relation_name]))
                    tokens.append(KeyToken(KeyType.Alias))
                    tokens.append(AliasToken(alias_idx))
                    if 'Recheck Cond' in node:
                        tokens.append(KeyToken(KeyType.RecheckCond))
                        tokens.extend(tokenizer(node['Recheck Cond'], plan_info, self.db_info, node, parent)[0])
                    if 'Filter' in node:
                        tokens.append(KeyToken(KeyType.Filter))
                        tokens.extend(tokenizer(node['Filter'], plan_info, self.db_info, node, parent)[0])
                case NodeType.BitmapIndexScan:
                    tokens.append(KeyToken(KeyType.IndexCond))
                    tokens.extend(tokenizer(node['Index Cond'], plan_info, self.db_info, node, parent)[0])
                case _:
                    raise NotImplementedError
            tokens.append(KeyToken(KeyType.PlanRows))
            normalized_rows = math.log(node['Plan Rows'] + 1) / math.log(self.max_card + 1)
            tokens.append(CardinalityToken(normalized_rows))
            token_end = len(tokens)
            node_card = node['Actual Rows'] * node['Actual Loops'] if 'Execution Time' in plan else -1.
            nodes.append(Node(token_begin, token_end, depth, node_begin, 0, leaf_begin, 0, node_card))
            node_prop = nodes[-1]
            node_end = node_begin + 1
            leaf_end = leaf_begin
            if 'Plans' in node:
                for child in node['Plans']:
                    node_end, leaf_end = dfs(child, node, depth + 1, node_end, leaf_end)
            else:
                leaf_end += 1
            node_prop.node_end = node_end
            node_prop.leaf_end = leaf_end
            return node_end, leaf_end

        root = plan['Plan']
        while 'Plans' in root and len(root['Plans']) == 1:
            root = root['Plans'][0]

        dfs(root)

        return Sample(tokens, nodes, plan.get('Execution Time', float('inf')), plan['Execution Time'] if 'Execution Time' in plan else plan['Timeout Time'], weight)