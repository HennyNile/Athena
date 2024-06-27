from enum import Enum
import sys

sys.path.append('.')
from src.utils.tokenizer import TokenType, tokenizer
from src.utils.db_utils import DBInfo

class DataType(Enum):
    Integer = 0
    Text    = 1

class OpType(Enum):
    Precise = 0
    Less    = 1
    Like    = 2

class Entity:
    def __init__(self):
        pass

class Column(Entity):
    def __init__(self, table: int, alias: int, column: int):
        super().__init__()
        self.table = table
        self.alias = alias
        self.column = column

    def __str__(self) -> str:
        return f'Column({self.table}, {self.alias}, {self.column})'

class Number(Entity):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return f'Number({self.value})'

class Array(Entity):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __str__(self) -> str:
        return f'Array({self.size})'
    
class Null(Entity):
    def __init__(self):
        super().__init__()
    
    def __str__(self) -> str:
        return f'NULL'

class Operator:
    def __init__(self, positive: bool, op: OpType):
        self.positive = positive
        self.op = op

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

class ExprTreeNode:
    def __init__(self):
        pass

class BoolOperatorNode(ExprTreeNode):
    def __init__(self, op: str, *nodes):
        self.op = op
        self.children_: list[ExprTreeNode] = nodes

    def children(self) -> list[ExprTreeNode]:
        return self.children_

    def __str__(self) -> str:
        return self.op

class AtomicNode(ExprTreeNode):
    def __init__(self, left: Column, op: Operator, right: Entity):
        self.left = left
        self.op = op
        self.right = right

    def children(self) -> list[ExprTreeNode]:
        return []

    def __str__(self) -> str:
        return f'{self.left} {self.op} {self.right}'

class ExprTree:
    def __init__(self, root):
        self.root = root

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
        self.idx = 0
        print(expr)

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

    def parse_expr(self) -> ExprTreeNode:
        ret = self.parse_term()
        while self.idx < len(self.tokens):
            t, token = self.tokens[self.idx]
            if t == TokenType.IDENTIFIER and token in ['AND', 'OR']:
                self.idx += 1
                another = self.parse_term()
                ret = BoolOperatorNode(token, ret, another)
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
        t, token = self.tokens[self.idx]
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
                    t, token = self.tokens[self.idx]
                    if t == TokenType.IDENTIFIER and token == 'NOT':
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
                self.idx += 1
                self.expect(TokenType.ATTRIBUTE)
                self.expect(TokenType.IDENTIFIER, ['text'])
                return Array(1)
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
        return Array(array_len)