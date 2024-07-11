from enum import Enum

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
    HASH          = 29

def tokenizer(expr: str) -> list[tuple[TokenType, str]]:
    tokens: list[tuple[TokenType, str]] = []
    idx = 0
    in_quote = False
    while idx < len(expr):
        if in_quote:
            if expr[idx] == '"':
                start = idx
                idx += 1
                while idx < len(expr) and expr[idx] != '"':
                    idx += 1
                if idx == len(expr):
                    raise ValueError(f'Invalid quote: {expr[start]} at {expr} : {start}')
                idx += 1
                tokens.append((TokenType.DOUBLE_QUOTE, expr[start:idx]))
            elif expr[idx] == ',':
                tokens.append((TokenType.COMMA, ','))
                idx += 1
            elif expr[idx] == '}':
                if idx + 1 < len(expr) and expr[idx + 1] == "'":
                    tokens.append((TokenType.ARRAY_END, "}'"))
                    idx += 2
                    in_quote = False
                else:
                    raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
            else:
                start = idx
                while idx < len(expr) and expr[idx] != ',' and expr[idx] != '}':
                    idx += 1
                tokens.append((TokenType.WORD, expr[start:idx]))
        elif expr[idx] == '(':
            tokens.append((TokenType.LEFT_PARAN, '('))
            idx += 1
        elif expr[idx] == ')':
            tokens.append((TokenType.RIGHT_PARAN, ')'))
            idx += 1
        elif expr[idx] == '[':
            tokens.append((TokenType.LEFT_BRACKET, '['))
            idx += 1
        elif expr[idx] == ']':
            tokens.append((TokenType.RIGHT_BRACKET, ']'))
            idx += 1
        elif expr[idx] == '{':
            tokens.append((TokenType.LEFT_CURLY, '{'))
            idx += 1
        elif expr[idx] == '}':
            tokens.append((TokenType.RIGHT_CURLY, '}'))
            idx += 1
        elif expr[idx] == '=':
            tokens.append((TokenType.EQUAL, '='))
            idx += 1
        elif expr[idx] == '<':
            if idx + 1 < len(expr) and (expr[idx + 1] == '=' or expr[idx + 1] == '>'):
                if expr[idx + 1] == '=':
                    tokens.append((TokenType.LESS_EQUAL, '<='))
                else:
                    tokens.append((TokenType.NOT_EQUAL, '<>'))
                idx += 2
            else:
                tokens.append((TokenType.LESS_THAN, '<'))
                idx += 1
        elif expr[idx] == '>':
            if idx + 1 < len(expr) and expr[idx + 1] == '=':
                tokens.append((TokenType.GREATER_EQUAL, '>='))
                idx += 2
            else:
                tokens.append((TokenType.GREATER_THAN, '>'))
                idx += 1
        elif expr[idx] == '!':
            if idx + 1 < len(expr) and expr[idx + 1] == '=':
                tokens.append((TokenType.NOT_EQUAL, '!='))
                idx += 2
            elif idx + 2 < len(expr) and expr[idx + 1] == '~' and expr[idx + 2] == '~':
                tokens.append((TokenType.NOT_LIKE, '!~~'))
                idx += 3
            else:
                raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
        elif expr[idx] == '~':
            if idx + 1 < len(expr) and expr[idx + 1] == '~':
                tokens.append((TokenType.LIKE, '~~'))
                idx += 2
            else:
                raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
        elif expr[idx] == '.':
            tokens.append((TokenType.DOT, '.'))
            idx += 1
        elif expr[idx] == ',':
            tokens.append((TokenType.COMMA, ','))
            idx += 1
        elif expr[idx] == "'":
            if idx + 1 < len(expr) and expr[idx + 1] == '{':
                tokens.append((TokenType.ARRAY_BEGIN, "'{"))
                idx += 2
                in_quote = True
            else:
                start = idx
                idx += 1
                while idx < len(expr) and expr[idx] != "'":
                    idx += 1
                if idx == len(expr):
                    raise ValueError(f'Invalid quote: {expr[start]} at {expr} : {start}')
                idx += 1
                tokens.append((TokenType.SINGLE_QUOTE, expr[start:idx]))
        elif expr[idx] == '"':
            start = idx
            idx += 1
            while idx < len(expr) and expr[idx] != '"':
                idx += 1
            if idx == len(expr):
                raise ValueError(f'Invalid quote: {expr[start]} at {expr} : {start}')
            idx += 1
            tokens.append((TokenType.DOUBLE_QUOTE, expr[start:idx]))
        elif expr[idx] == '%':
            tokens.append((TokenType.WILDCARD, '%'))
            idx += 1
        elif expr[idx] == ':':
            if idx + 1 < len(expr) and expr[idx + 1] == ':':
                tokens.append((TokenType.ATTRIBUTE, '::'))
                idx += 2
            else:
                raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
        elif expr[idx].isalpha():
            start = idx
            idx += 1
            while idx < len(expr) and (expr[idx].isalnum() or expr[idx] == '_' or expr[idx] == '-'):
                idx += 1
            tokens.append((TokenType.IDENTIFIER, expr[start:idx]))
        elif expr[idx].isdigit():
            start = idx
            idx += 1
            while idx < len(expr) and (expr[idx].isdigit() or expr[idx] == '.'):
                idx += 1
            tokens.append((TokenType.NUMBER, expr[start:idx]))
        elif expr[idx].isspace():
            idx += 1
        else:
            raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
    return tokens

def string_splitter(expr: str) -> list[tuple[TokenType, str]]:
    tokens: list[tuple[TokenType, str]] = []
    idx = 0
    while idx < len(expr):
        if expr[idx] == '(':
            tokens.append((TokenType.LEFT_PARAN, '('))
            idx += 1
        elif expr[idx] == ')':
            tokens.append((TokenType.RIGHT_PARAN, ')'))
            idx += 1
        elif expr[idx] == '[':
            tokens.append((TokenType.LEFT_BRACKET, '['))
            idx += 1
        elif expr[idx] == ']':
            tokens.append((TokenType.RIGHT_BRACKET, ']'))
            idx += 1
        elif expr[idx] == '%':
            tokens.append((TokenType.WILDCARD, '%'))
            idx += 1
        elif expr[idx] == '-':
            tokens.append((TokenType.HYPHEN, '-'))
            idx += 1
        elif expr[idx] == '_':
            tokens.append((TokenType.UNDER_SCORE, '_'))
            idx += 1
        elif expr[idx] == ':':
            tokens.append((TokenType.COLON, ':'))
            idx += 1
        elif expr[idx] == '+':
            tokens.append((TokenType.ADD, '+'))
            idx += 1
        elif expr[idx] == '.':
            tokens.append((TokenType.DOT, '.'))
            idx += 1
        elif expr[idx] == ',':
            tokens.append((TokenType.COMMA, ','))
            idx += 1
        elif expr[idx] == '#':
            tokens.append((TokenType.HASH, '#'))
            idx += 1
        elif expr[idx] == '>':
            tokens.append((TokenType.GREATER_THAN, '>'))
            idx += 1
        elif expr[idx].isalpha():
            start = idx
            idx += 1
            while idx < len(expr) and expr[idx].isalpha():
                idx += 1
            tokens.append((TokenType.WORD, expr[start:idx]))
        elif expr[idx].isdigit():
            start = idx
            idx += 1
            while idx < len(expr) and (expr[idx].isdigit() or expr[idx] == '.'):
                idx += 1
            tokens.append((TokenType.WORD, expr[start:idx]))
        elif expr[idx].isspace():
            idx += 1
        else:
            raise ValueError(f'Invalid character: {expr[idx]} at {expr} : {idx}')
    return tokens