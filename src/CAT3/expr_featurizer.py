from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
import sys

import numpy as np

sys.path.append('.')
from src.utils.tokenizer import TokenType, tokenizer
from src.utils.db_utils import DBInfo

class LogicalType(Enum):
    And = 0
    Or  = 1

class OpType(Enum):
    Precise = 0
    Less    = 1
    Like    = 2

class Entity(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __eq__(self, value: 'Entity') -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

class Column(Entity):
    def __init__(self, table: int, alias: int, column: int):
        super().__init__()
        self.table = table
        self.alias = alias
        self.column = column

    def __eq__(self, value: Entity) -> bool:
        if type(value) != Column:
            return False
        return value.table == self.table and value.alias == self.alias and value.column == self.column

    def __str__(self) -> str:
        return f'Column({self.table}, {self.alias}, {self.column})'

class Number(Entity):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __eq__(self, value: Entity) -> bool:
        if type(value) != Number:
            return False
        return value.value == self.value

    def __str__(self) -> str:
        return f'Number({self.value})'

class Timestamp(Entity):
    def __init__(self, repr: str):
        super().__init__()
        self.timestamp = datetime.strptime(repr, '%Y-%m-%d %H:%M:%S')

    def __eq__(self, value: Entity) -> bool:
        if type(value) != Timestamp:
            return False
        return value.timestamp == self.timestamp
    
    def __str__(self) -> str:
        return f'Timestamp({self.timestamp})'

class Array(Entity):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __eq__(self, value: Entity) -> bool:
        if type(value) != Array:
            return False
        return value.size == self.size

    def __str__(self) -> str:
        return f'Array({self.size})'

class Null(Entity):
    def __init__(self):
        super().__init__()

    def __eq__(self, value: Entity) -> bool:
        return type(value) == Null
    
    def __str__(self) -> str:
        return f'NULL'

class Operator:
    def __init__(self, positive: bool, op: OpType):
        self.positive = positive
        self.op = op

    def __eq__(self, value: 'Operator') -> bool:
        return value.positive == self.positive and value.op == self.op

    def __str__(self) -> str:
        match self.op:
            case OpType.Precise:
                if self.positive:
                    return '='
                else:
                    return '!='
            case OpType.Less:
                if self.positive:
                    return '<'
                else:
                    return '>='
            case OpType.Like:
                if self.positive:
                    return 'LIKE'
                else:
                    return 'NOT LIKE'

class ExprTreeNode(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def children(self) -> list['ExprTreeNode']:
        pass

    @abstractmethod
    def __eq__(self, value: 'ExprTreeNode') -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

class BoolOperatorNode(ExprTreeNode):
    def __init__(self, op: LogicalType, *nodes):
        self.op = op
        self.children_: list[ExprTreeNode] = list(nodes)

    def children(self) -> list[ExprTreeNode]:
        return self.children_

    def __eq__(self, value: ExprTreeNode) -> bool:
        if type(value) != BoolOperatorNode:
            return False
        if value.op != self.op:
            return False
        if len(value.children_) != len(self.children_):
            return False
        for other, child in zip(value.children_, self.children_):
            if other != child:
                return False
        return True

    def __str__(self) -> str:
        match self.op:
            case LogicalType.And:
                return 'AND'
            case LogicalType.Or:
                return 'OR'

class AtomicNode(ExprTreeNode):
    def __init__(self, left: Column, op: Operator, right: Entity):
        self.left = left
        self.op = op
        self.right = right

    def children(self) -> list[ExprTreeNode]:
        return []
    
    def __eq__(self, value: ExprTreeNode) -> bool:
        if type(value) != AtomicNode:
            return False
        return value.left == self.left and value.op == self.op and value.right == self.right

    def __str__(self) -> str:
        return f'{self.left} {self.op} {self.right}'

class IndexTreeNode:
    def __init__(self, t: LogicalType):
        self.op = t
        self.children = []

class IndexTree:
    def __init__(self, root: IndexTreeNode|int):
        self.root = root

    def pool_once(self, op: LogicalType):
        indices: list[int] = []
        def dfs(node: IndexTreeNode, parent: IndexTreeNode|None = None, child_idx: int|None = None, begin: int = 1):
            all_leaves = False
            if node.op == op:
                if len(node.children) != 2:
                    raise RuntimeError("Invalid node")
                if type(node.children[0]) == int and type(node.children[1]) == int:
                    all_leaves = True
                    indices.extend(node.children)
                    if parent is not None:
                        parent.children[child_idx] = begin
                    else:
                        self.root = begin
                    begin += 1
            if not all_leaves:
                for i, child in enumerate(node.children):
                    if type(child) == int:
                        indices.extend([child, 0])
                        node.children[i] = begin
                        begin += 1
                    else:
                        begin = dfs(child, node, i, begin)
            return begin
        dfs(self.root)
        return indices

    def pool(self):
        ret: list[list[int]] = []
        while type(self.root) != int:
            indices = self.pool_once(LogicalType.And if len(ret) % 2 == 0 else LogicalType.Or)
            ret.append(indices)
        return ret

class ExprTree:
    def __init__(self, root):
        self.root = root
        self.vecs: np.ndarray|None = None
        self.indices: list[list[int]]|None = None

    def index_tree(self, idx_begin: int = 1):
        def dfs(node):
            if type(node) == BoolOperatorNode:
                ret = IndexTreeNode(node.op)
                for child in node.children():
                    ret.children.append(dfs(child))
                return ret
            elif type(node) == AtomicNode:
                nonlocal idx_begin
                ret = idx_begin
                idx_begin += 1
                return ret
            else:
                raise RuntimeError("Wrong node type")
        root = dfs(self.root)
        return IndexTree(root), idx_begin

    def cache_features(self, featurize_node: Callable[[AtomicNode], np.ndarray]) -> None:
        vecs = []
        def dfs(node):
            if type(node) == BoolOperatorNode:
                for child in node.children():
                    dfs(child)
            elif type(node) == AtomicNode:
                vecs.append(featurize_node(node))
            else:
                raise RuntimeError("Wrong node type")
        dfs(self.root)
        self.vecs = np.stack(vecs)
        index_tree, _ = self.index_tree()
        self.indices = index_tree.pool()

    def combine_disjunction(self):
        changed = True
        while changed:
            changed = False
            def dfs(node: ExprTreeNode, parent: BoolOperatorNode|None = None, child_idx: int|None = None):
                nonlocal changed
                children = node.children()
                if type(node) == BoolOperatorNode and node.op == LogicalType.Or and len(node.children()) > 0:
                    first = children[0]
                    if type(first) == AtomicNode and first.op.positive and type(first.right) == Array:
                        all_same = True
                        for child in children[1:]:
                            if type(child) != AtomicNode or child.left != first.left or child.op != first.op or type(child.right) != Array:
                                all_same = False
                                break
                        if all_same:
                            new_node = AtomicNode(first.left, first.op, Array(sum([child.right.size for child in children])))
                            if parent is not None:
                                parent.children_[child_idx] = new_node
                            else:
                                self.root = new_node
                            changed = True
                            return
                for i, child in enumerate(children):
                    dfs(child, node, i)
            dfs(self.root)

    def __str__(self) -> str:
        lines = []
        def recurse(node: ExprTreeNode, is_last: list[bool]):
            indent = ''
            if len(is_last) > 0:
                for last in is_last[:-1]:
                    indent += '  ' if last else '│ '
                indent += '└─' if is_last[-1] else '├─'
            nonlocal lines
            lines.append(f'{indent}{node}')
            for i, child in enumerate(node.children()):
                recurse(child, is_last + [i == len(node.children()) - 1])
        recurse(self.root, [])
        return '\n'.join(lines)

class PlanInfo:
    def __init__(self, alias_map: dict[str, tuple[str, int]], rel_names: set[str]) -> None:
        self.alias_map = alias_map
        self.rel_names = rel_names

class ExprParser:
    def __init__(self, expr: str, plan_info: PlanInfo, db_info: DBInfo, node: dict, node_parent: dict|None) -> None:
        self.plan_info = plan_info
        self.db_info = db_info
        self.node = node
        self.node_parent = node_parent
        self.tokens = tokenizer(expr)
        self.max_alias_idx = 0
        self.max_array_len = 1
        self.idx = 0

    def __call__(self) -> ExprTree:
        ret = ExprTree(self.parse_expr())
        return ret

    def expect(self, ts, tokens = None):
        next_t, next_token = self.tokens[self.idx]
        if type(ts) == list:
            for t, token in zip(ts, tokens):
                if t == next_t and (token is None or next_token in token):
                    self.idx += 1
                    return next_t, next_token
            raise RuntimeError(f'Unexpected token: {next_t}, {next_token} at {self.idx}')
        else:
            if ts == next_t and (tokens is None or next_token in tokens):
                self.idx += 1
                return next_t, next_token
            raise RuntimeError(f'Unexpected token: {next_t}, {next_token} at {self.idx}')

    # <expr>        :== <term> [("AND" | "OR") <expr>]
    # <term>        :== "(" <expr> ")" | <column> <op> <entity>
    # <entity>      :== <column> | <value> | <array> | "NULL" | <text> | <time> | <integer>
    # <column>      :== "(" <column name> ")" "::" "text" | <column name>
    # <column name> :== identifier "." identifier | identifier
    # <op>          :== "=" | "!=" | "<" | "<=" | ">" | ">=" | "~~" | "!~~" | "IS" | "IS NOT"
    # <value>       :== number
    # <array>       :== "ANY" "(" "'{" [<array list>] "}'" "::" "text" "[" "]" ")"
    # <array list>  :== (word | double_quote | single_quote) ["," <array list>]
    # <text>        :== single_quote "::" "text"
    # <integer>     :== single_quote "::" "integer"
    # <time>        :== single_quote "::" "timestamp" "without" "time" "zone"

    def parse_expr(self) -> ExprTreeNode:
        ret = self.parse_term()
        while self.idx < len(self.tokens):
            t, token = self.tokens[self.idx]
            if t == TokenType.IDENTIFIER and token in ['AND', 'OR']:
                self.idx += 1
                another = self.parse_term()
                ret = BoolOperatorNode(LogicalType.And if token == 'AND' else LogicalType.Or, ret, another)
            else:
                break
        return ret

    def parse_term(self) -> AtomicNode:
        t, _ = self.tokens[self.idx]
        is_expr = True
        match t:
            case TokenType.LEFT_PARAN:
                current = self.idx + 1
                while current < len(self.tokens):
                    t, _ = self.tokens[current]
                    if t == TokenType.LEFT_PARAN:
                        break
                    if t == TokenType.RIGHT_PARAN:
                        if current + 1 < len(self.tokens) and self.tokens[current + 1][0] == TokenType.ATTRIBUTE:
                            is_expr = False
                        break
                    current += 1
            case TokenType.IDENTIFIER:
                is_expr = False
        if is_expr:
            self.idx += 1
            ret = self.parse_expr()
            self.expect(TokenType.RIGHT_PARAN)
            return ret
        else:
            column = self.parse_column()
            op, delta = self.parse_op()
            entity = self.parse_entity(column.column, delta)

            return AtomicNode(column, op, entity)

    def parse_column(self) -> Column:
        t, _ = self.tokens[self.idx]
        match t:
            case TokenType.LEFT_PARAN:
                self.idx += 1
                ret = self.parse_column_name()
                self.expect(TokenType.RIGHT_PARAN)
                self.expect(TokenType.ATTRIBUTE)
                self.expect(TokenType.IDENTIFIER, ['text'])
                return ret
            case _:
                return self.parse_column_name()

    def parse_column_name(self) -> Column:
        _, first_name = self.expect(TokenType.IDENTIFIER)
        t, _ = self.tokens[self.idx]
        if t == TokenType.DOT:
            self.idx += 1
            _, second_name = self.expect(TokenType.IDENTIFIER)
            if first_name in self.plan_info.alias_map:
                table_name, alias_idx = self.plan_info.alias_map[first_name]
                table_idx = self.db_info.table_map[table_name]
            elif first_name in self.plan_info.rel_names:
                table_idx = self.db_info.table_map[first_name]
                alias_idx = 0
            else:
                raise RuntimeError(f'Unknown alias name: {first_name}')
            column_idx = self.db_info.column_map_list[table_idx][second_name]
            return Column(table_idx, alias_idx, column_idx)
        else:
            if 'Alias' in self.node:
                alias_name = self.node['Alias']
                table_name, alias_idx = self.plan_info.alias_map[alias_name]
                table_idx = self.db_info.table_map[table_name]
            elif 'Relation Name' in self.node:
                table_name = self.node['Relation Name']
                table_idx = self.db_info.table_map[table_name]
                alias_idx = 0
            elif self.node_parent is not None and 'Alias' in self.node_parent:
                alias_name = self.node_parent['Alias']
                table_name, alias_idx = self.plan_info.alias_map[alias_name]
                table_idx = self.db_info.table_map[table_name]
            elif self.node_parent is not None and 'Relation Name' in self.node_parent:
                table_name = self.node_parent['Relation Name']
                table_idx = self.db_info.table_map[table_name]
                alias_idx = 0
            else:
                raise RuntimeError(f'Unknown alias name for column: {first_name}')
            column_idx = self.db_info.column_map_list[table_idx][first_name]
            self.max_alias_idx = max(self.max_alias_idx, alias_idx)
            return Column(table_idx, alias_idx, column_idx)

    def parse_op(self) -> tuple[Operator, float]:
        t, token = self.tokens[self.idx]
        self.idx += 1
        delta = 0
        match t:
            case TokenType.EQUAL:
                ret = Operator(True, OpType.Precise)
            case TokenType.NOT_EQUAL:
                ret = Operator(False, OpType.Precise)
            case TokenType.LESS_THAN:
                ret = Operator(True, OpType.Less)
            case TokenType.LESS_EQUAL:
                delta = 1
                ret = Operator(True, OpType.Less)
            case TokenType.GREATER_THAN:
                delta = 1
                ret = Operator(False, OpType.Less)
            case TokenType.GREATER_EQUAL:
                ret = Operator(False, OpType.Less)
            case TokenType.LIKE:
                ret = Operator(True, OpType.Like)
            case TokenType.NOT_LIKE:
                ret = Operator(False, OpType.Like)
            case TokenType.IDENTIFIER:
                if token == 'IS':
                    if self.idx < len(self.tokens) and self.tokens[self.idx] == (TokenType.IDENTIFIER, 'NOT'):
                        self.idx += 1
                        ret = Operator(False, OpType.Precise)
                    else:
                        ret = Operator(True, OpType.Precise)
                else:
                    raise RuntimeError(f'Unexpected identifier: {token} at {self.idx - 1}')
            case _:
                raise RuntimeError(f'Unexpected token: {t}, {token} at {self.idx - 1}')
        return ret, delta

    def parse_entity(self, column_idx: int, delta: float) -> Entity:
        t, token = self.tokens[self.idx]
        match t:
            case TokenType.NUMBER:
                return self.parse_value(column_idx, delta)
            case TokenType.IDENTIFIER:
                if token == 'ANY':
                    return self.parse_array()
                elif token == 'NULL':
                    self.idx += 1
                    return Null()
                else:
                    return self.parse_column()
            case TokenType.SINGLE_QUOTE:
                if self.tokens[self.idx + 1][0] != TokenType.ATTRIBUTE:
                    raise RuntimeError(f'Unexpected token: {t}, {token} at {self.idx + 1}')
                if self.tokens[self.idx + 2][0] != TokenType.IDENTIFIER or self.tokens[self.idx + 2][1] not in ('text', 'timestamp', 'integer'):
                    raise RuntimeError(f'Unexpected token: {t}, {token} at {self.idx + 2}')
                if self.tokens[self.idx + 2][1] == 'integer':
                    return self.parse_integer(column_idx, delta)
                elif self.tokens[self.idx + 2][1] == 'text':
                    return self.parse_text()
                else:
                    return self.parse_timestamp()
            case _:
                raise RuntimeError(f'Unexpected token: {t}, {token} at {self.idx}')

    def parse_value(self, column_idx: int, delta: float) -> Number:
        _, token = self.tokens[self.idx]
        self.idx += 1
        m, M = self.db_info.get_normalizer(column_idx)
        value = float(token) + delta
        value = (value - m) / (M - m)
        return Number(value)

    def parse_array(self) -> Array:
        self.expect(TokenType.IDENTIFIER, ['ANY'])
        self.expect(TokenType.LEFT_PARAN)
        self.expect(TokenType.ARRAY_BEGIN)
        t, token = self.tokens[self.idx]
        if t == TokenType.ARRAY_END:
            self.idx += 1
            self.expect(TokenType.ATTRIBUTE)
            self.expect(TokenType.IDENTIFIER, ['text'])
            self.expect(TokenType.LEFT_BRACKET)
            self.expect(TokenType.RIGHT_BRACKET)
            self.expect(TokenType.RIGHT_PARAN)
            return Array(0)
        array_len = 0
        while True:
            t, _ = self.expect([TokenType.SINGLE_QUOTE, TokenType.DOUBLE_QUOTE, TokenType.WORD], [None, None, None])
            array_len += 1
            t, _ = self.expect([TokenType.COMMA, TokenType.ARRAY_END], [None, None])
            if t == TokenType.ARRAY_END:
                break
        self.expect(TokenType.ATTRIBUTE)
        self.expect(TokenType.IDENTIFIER, ['text'])
        self.expect(TokenType.LEFT_BRACKET)
        self.expect(TokenType.RIGHT_BRACKET)
        self.expect(TokenType.RIGHT_PARAN)
        self.max_array_len = max(self.max_array_len, array_len)
        return Array(array_len)

    def parse_integer(self, column_idx: int, delta: float) -> Number:
        _, token = self.tokens[self.idx]
        self.idx += 3
        m, M = self.db_info.get_normalizer(column_idx)
        value = float(token[1:-1]) + delta
        value = (value - m) / (M - m)
        return Number(value)

    def parse_text(self) -> Array:
        self.idx += 3
        return Array(1)

    def parse_timestamp(self) -> Array:
        _, timestamp_str = self.tokens[self.idx]
        self.idx += 3
        self.expect(TokenType.IDENTIFIER, 'without')
        self.expect(TokenType.IDENTIFIER, 'time')
        self.expect(TokenType.IDENTIFIER, 'zone')
        return Timestamp(timestamp_str[1:-1])