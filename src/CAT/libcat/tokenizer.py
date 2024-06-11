from enum import Enum

import torch

import plan_expr_parser

class TokenType(Enum):
    LEFT_PARAN    = 0
    RIGHT_PARAN   = 1
    LEFT_BRACKET  = 2
    RIGHT_BRACKET = 3
    LEFT_CURLY    = 4
    RIGHT_CURLY   = 5
    ARRAY_BEGIN   = 6
    ARRAY_END     = 7
    EQUAL         = 8
    LESS_THAN     = 9
    GREATER_THAN  = 10
    LESS_EQUAL    = 11
    GREATER_EQUAL = 12
    NOT_EQUAL     = 13
    LIKE          = 14
    NOT_LIKE      = 15
    DOT           = 16
    COMMA         = 17
    HYPHEN        = 18
    UNDER_SCORE   = 19
    COLON         = 20
    ADD           = 21
    SINGLE_QUOTE  = 22
    DOUBLE_QUOTE  = 23
    WILDCARD      = 24
    ATTRIBUTE     = 25
    IDENTIFIER    = 26
    NUMBER        = 27
    WORD          = 28

class InputTokenType(Enum):
    SYNTAX    = 0
    TABLE     = 1
    TABLE_IDX = 2
    COLUMN    = 3
    WORD      = 4
    NUMBER    = 5

class SyntaxType(Enum):
    LEFT_PARAN    = 0
    RIGHT_PARAN   = 1
    LEFT_BRACKET  = 2
    RIGHT_BRACKET = 3
    ARRAY_BEGIN   = 4
    ARRAY_END     = 5
    EQUAL         = 6
    LESS_THAN     = 7
    GREATER_THAN  = 8
    LESS_EQUAL    = 9
    GREATER_EQUAL = 10
    NOT_EQUAL     = 11
    DOT           = 12
    COMMA         = 13
    SINGLE_QUOTE  = 14
    DOUBLE_QUOTE  = 15
    WILDCARD      = 16
    ATTRIBUTE     = 17
    OP_LIKE       = 18
    OP_NOT_LIKE   = 19
    KEYWORD_TEXT  = 20
    KEYWORD_ANY   = 21
    KEYWORD_AND   = 22
    KEYWORD_OR    = 23
    KEYWORD_IS    = 24
    KEYWORD_NOT   = 25
    KEYWORD_NULL  = 26

def word_splitter(word):
    word_types, words = plan_expr_parser.string_splitter(word)
    word_types = [TokenType(w) for w in word_types]
    ret = []
    for wt, word in zip(word_types, words):
        match wt:
            case TokenType.WORD:
                ret.append((InputTokenType.WORD, word))
            case TokenType.LEFT_PARAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.LEFT_PARAN))
            case TokenType.RIGHT_PARAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.RIGHT_PARAN))
            case TokenType.LEFT_BRACKET:
                ret.append((InputTokenType.WORD, '['))
            case TokenType.RIGHT_BRACKET:
                ret.append((InputTokenType.WORD, ']'))
            case TokenType.WILDCARD:
                ret.append((InputTokenType.SYNTAX, SyntaxType.WILDCARD))
            case TokenType.COMMA:
                ret.append((InputTokenType.WORD, ','))
            case TokenType.HYPHEN:
                ret.append((InputTokenType.WORD, '-'))
            case TokenType.UNDER_SCORE:
                ret.append((InputTokenType.WORD, '_'))
            case TokenType.COLON:
                ret.append((InputTokenType.WORD, ':'))
            case TokenType.ADD:
                ret.append((InputTokenType.WORD, '+'))
            case _:
                raise NotImplementedError
    return ret

def tokenizer(expr: str, alias_map: dict[str, str], rel_names: set[str], node: dict, node_parent: dict|None) -> tuple[list[tuple[InputTokenType, any]], int]:
    max_table_idx = 0
    token_types, tokens = plan_expr_parser.tokenizer(expr)
    token_types = [TokenType(t) for t in token_types]
    ret = []
    for idx, (t, token) in enumerate(zip(token_types, tokens)):
        match t:
            case TokenType.LEFT_PARAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.LEFT_PARAN))
            case TokenType.RIGHT_PARAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.RIGHT_PARAN))
            case TokenType.LEFT_BRACKET:
                ret.append((InputTokenType.SYNTAX, SyntaxType.LEFT_BRACKET))
            case TokenType.RIGHT_BRACKET:
                ret.append((InputTokenType.SYNTAX, SyntaxType.RIGHT_BRACKET))
            case TokenType.ARRAY_BEGIN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.ARRAY_BEGIN))
            case TokenType.ARRAY_END:
                ret.append((InputTokenType.SYNTAX, SyntaxType.ARRAY_END))
            case TokenType.EQUAL:
                ret.append((InputTokenType.SYNTAX, SyntaxType.EQUAL))
            case TokenType.LESS_THAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.LESS_THAN))
            case TokenType.GREATER_THAN:
                ret.append((InputTokenType.SYNTAX, SyntaxType.GREATER_THAN))
            case TokenType.LESS_EQUAL:
                ret.append((InputTokenType.SYNTAX, SyntaxType.LESS_EQUAL))
            case TokenType.GREATER_EQUAL:
                ret.append((InputTokenType.SYNTAX, SyntaxType.GREATER_EQUAL))
            case TokenType.NOT_EQUAL:
                ret.append((InputTokenType.SYNTAX, SyntaxType.NOT_EQUAL))
            case TokenType.DOT:
                ret.append((InputTokenType.SYNTAX, SyntaxType.DOT))
            case TokenType.COMMA:
                ret.append((InputTokenType.SYNTAX, SyntaxType.COMMA))
            case TokenType.ATTRIBUTE:
                ret.append((InputTokenType.SYNTAX, SyntaxType.ATTRIBUTE))
            case TokenType.LIKE:
                ret.append((InputTokenType.SYNTAX, SyntaxType.OP_LIKE))
            case TokenType.NOT_LIKE:
                ret.append((InputTokenType.SYNTAX, SyntaxType.OP_NOT_LIKE))
            case TokenType.SINGLE_QUOTE:
                ret.append((InputTokenType.SYNTAX, SyntaxType.SINGLE_QUOTE))
                ret.extend(word_splitter(token[1:-1]))
                ret.append((InputTokenType.SYNTAX, SyntaxType.SINGLE_QUOTE))
            case TokenType.DOUBLE_QUOTE:
                ret.append((InputTokenType.SYNTAX, SyntaxType.DOUBLE_QUOTE))
                ret.extend(word_splitter(token[1:-1]))
                ret.append((InputTokenType.SYNTAX, SyntaxType.DOUBLE_QUOTE))
            case TokenType.IDENTIFIER:
                if token in alias_map:
                    table_name, table_idx = alias_map[token]
                    ret.append((InputTokenType.TABLE, table_name))
                    ret.append((InputTokenType.TABLE_IDX, table_idx))
                    max_table_idx = max(max_table_idx, table_idx)
                elif token in rel_names:
                    ret.append((InputTokenType.TABLE, token))
                    ret.append((InputTokenType.TABLE_IDX, 0))
                elif idx > 0 and (token_types[idx - 1] == TokenType.DOT or token_types[idx - 1] == TokenType.LEFT_PARAN):
                    if len(ret) >= 3 and ret[-1] == (InputTokenType.SYNTAX, SyntaxType.DOT) and ret[-3][0] == InputTokenType.TABLE:
                        table_name = ret[-3][1]
                    elif 'Relation Name' in node:
                        table_name = node['Relation Name']
                    elif node_parent is not None and 'Relation Name' in node_parent:
                        table_name = node_parent['Relation Name']
                    else:
                        raise RuntimeError("Unknown table name")
                    ret.append((InputTokenType.COLUMN, (table_name, token)))
                elif token == 'text':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_TEXT))
                elif token == 'ANY':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_ANY))
                elif token == 'AND':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_AND))
                elif token == 'OR':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_OR))
                elif token == 'IS':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_IS))
                elif token == 'NOT':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_NOT))
                elif token == 'NULL':
                    ret.append((InputTokenType.SYNTAX, SyntaxType.KEYWORD_NULL))
                else:
                    raise NotImplementedError
            case TokenType.NUMBER:
                if len(ret) >= 2 and ret[-2][0] == InputTokenType.COLUMN:
                    ret.append((InputTokenType.NUMBER, (ret[-2][1], float(token))))
            case TokenType.WORD:
                ret.extend(word_splitter(token))
            case _:
                raise NotImplementedError
    return ret, max_table_idx

def normalize_number(normalizer, key, value):
    min_v, max_v = normalizer[key]
    return (value - min_v) / (max_v - min_v)

def vectorize_tokens(tokens, num_input_types, dim, table_map, column_map, word_table, normalizer) -> torch.Tensor:
    seq = torch.zeros((len(tokens), dim), dtype=torch.float)
    for idx, (t, token) in enumerate(tokens):
        seq[idx, t.value] = 1.
        match t:
            case InputTokenType.SYNTAX:
                seq[idx, num_input_types + token.value] = 1.
            case InputTokenType.TABLE:
                seq[idx, num_input_types + table_map[token]] = 1.
            case InputTokenType.TABLE_IDX:
                seq[idx, num_input_types + token] = 1.
            case InputTokenType.COLUMN:
                seq[idx, num_input_types + column_map[token]] = 1.
            case InputTokenType.WORD:
                seq[idx, num_input_types + word_table[token]] = 1.
            case InputTokenType.NUMBER:
                seq[idx, num_input_types] = normalize_number(normalizer, *token)
    return seq

def featurize_exprs(sample: dict, expr_id: int, alias_map: dict[str, str], rel_names: set[str]) -> list[list[tuple[InputTokenType, any]]]:
    root = sample['Plan']
    token_batches = []
    num_nodes = 0

    def dfs(node, parent=None):
        nonlocal expr_id
        nonlocal num_nodes
        num_nodes += 1
        if 'Hash Cond' in node:
            token_batches.append(tokenizer(node['Hash Cond'], alias_map, rel_names, node, parent)[0])
            node['Hash Cond'] = expr_id
            expr_id += 1
        if 'Index Cond' in node:
            token_batches.append(tokenizer(node['Index Cond'], alias_map, rel_names, node, parent)[0])
            node['Index Cond'] = expr_id
            expr_id += 1
        if 'Recheck Cond' in node:
            token_batches.append(tokenizer(node['Recheck Cond'], alias_map, rel_names, node, parent)[0])
            node['Recheck Cond'] = expr_id
            expr_id += 1
        if 'Merge Cond' in node:
            token_batches.append(tokenizer(node['Merge Cond'], alias_map, rel_names, node, parent)[0])
            node['Merge Cond'] = expr_id
            expr_id += 1
        if 'Filter' in node:
            token_batches.append(tokenizer(node['Filter'], alias_map, rel_names, node, parent)[0])
            node['Filter'] = expr_id
            expr_id += 1
        if 'Join Filter' in node:
            token_batches.append(tokenizer(node['Join Filter'], alias_map, rel_names, node, parent)[0])
            node['Join Filter'] = expr_id
            expr_id += 1
        for child in node.get('Plans', []):
            dfs(child, node)

    dfs(root)
    return token_batches, num_nodes