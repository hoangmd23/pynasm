from enum import StrEnum
from typing import NamedTuple


class LexerError(Exception):
    pass


KEYWORDS = {
    # sections
    'section',
    'data',
    'text',
    'db',
    # unops
    'dec',
    # binops
    'mov',
    'add',
    'xor',
    'cmp',
    # jumps
    'jne',
    # registers
    'rax',
    'rbx',
    'rdx',
    'rsi',
    'rsp',
    'rbp',
    'rcx',
    'rdi',
    'eax',
    # stack
    'push',
    'pop',
}


class TokenType(StrEnum):
    Number = 'Number'
    String = 'String'
    Identifier = 'Identifier'
    Keyword = 'Keyword'
    OpeningSquareBracket = 'OpeningSquareBracket'
    ClosingSquareBracket = 'ClosingSquareBracket'
    Operator = 'Operator'
    Comma = 'Comma'
    Dot = 'Dot'
    Colon = 'Colon'
    NewLine = 'NewLine'


class Token(NamedTuple):
    type: TokenType
    value: str | None

    def __str__(self):
        return f'{str(self.type)}({self.value if self.value is not None else ''})'


class Lexer:
    def __init__(self, code: str, debug: bool = False):
        self.code: str = code
        self.debug: bool = debug
        self.pos: int = 0

    def parse_next_token(self, code: str, pos: int) -> tuple[Token | None, int]:
        if pos >= len(code):
            return None, pos

        while pos < len(code) and code[pos] == ' ':
            pos += 1

        if code[pos] == ';':
            # skip comment
            while pos < len(code) and code[pos] != '\n':
                pos += 1

        if code[pos] == '\n':
            # newline
            token_type = TokenType.NewLine
            token_value = None
            pos += 1
        elif code[pos].isdigit():
            # number
            token_type = TokenType.Number
            token_value = [code[pos]]
            pos += 1
            while pos < len(code) and code[pos].isdigit():
                token_value.append(code[pos])
                pos += 1
            token_value = ''.join(token_value)
        elif code[pos].isidentifier():
            # identifier
            token_value = [code[pos]]
            pos += 1
            while pos < len(code) and code[pos].isidentifier():
                token_value.append(code[pos])
                pos += 1
            token_value = ''.join(token_value)
            if token_value in KEYWORDS:
                token_type = TokenType.Keyword
            else:
                token_type = TokenType.Identifier
        elif code[pos] == '[':
            token_type = TokenType.OpeningSquareBracket
            token_value = code[pos]
            pos += 1
        elif code[pos] == ']':
            token_type = TokenType.ClosingSquareBracket
            token_value = code[pos]
            pos += 1
        elif code[pos] == ',':
            token_type = TokenType.Comma
            token_value = code[pos]
            pos += 1
        elif code[pos] == ':':
            token_type = TokenType.Colon
            token_value = None
            pos += 1
        elif code[pos] in '+-*<>':
            token_type = TokenType.Operator
            token_value = code[pos]
            pos += 1
        elif code[pos] == '.':
            token_type = TokenType.Dot
            token_value = code[pos]
            pos += 1
        elif code[pos] in '"\'':
            # parse string
            token_type = TokenType.String
            quote = code[pos]
            start = pos + 1
            pos += 1
            while code[pos] != quote:
                pos += 1
            token_value = code[start:pos]
            pos += 1
        else:
            raise NotImplementedError(f'Unknown token: {code[pos]}, {ord(code[pos])}')

        return Token(type=token_type, value=token_value), pos

    def _peek(self) -> tuple[Token | None, int]:
        token, pos = self.parse_next_token(self.code, self.pos)
        return token, pos

    def next_or_none(self) -> Token | None:
        token, self.pos = self._peek()
        if self.debug:
            print(f'Token: {token}')
        return token

    def next(self) -> Token:
        token = self.next_or_none()
        if token is None:
            raise LexerError(f'Could not get next token: there is no more tokens')
        return token

    def peek(self) -> Token | None:
        token, _ = self._peek()
        return token

    def expect(self, *token_types: TokenType, value=None) -> Token:
        token = self.next()
        if token is None:
            raise LexerError(f"Could not get next token: there is no more tokens")
        elif token.type in token_types:
            if value is not None and token.value != value:
                raise LexerError(f"Expected value: {value}, but got ({token.value})")
            else:
                return token
        else:
            raise LexerError(f"Expected token types: {token_types}, but got {token.type} ({token.value})")
