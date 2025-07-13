from dataclasses import dataclass
from enum import StrEnum

from nasmvis.common import Register
from nasmvis.lexer import Lexer, TokenType


class ParserError(Exception):
    pass


class InstType(StrEnum):
    mov = 'mov'
    add = 'add'

type Operand = Register | int

@dataclass
class Inst:
    line: int
    type: InstType
    operands: list[Operand] | None = None


def parse_mov(lexer: Lexer, line: int) -> Inst:
    # parse destination
    dest = lexer.expect(TokenType.Keyword).value
    if dest not in Register:
        raise ParserError(f'{line}: Unexpected destination operand {dest}')
    dest_op = Register(dest)

    # parse comma
    lexer.expect(TokenType.Comma)

    # parse source
    src = lexer.expect(TokenType.Number).value
    src_op = int(src)

    return Inst(line, InstType.mov, [dest_op, src_op])


def parse_add(lexer: Lexer, line: int) -> Inst:
    # parse destination
    dest = lexer.expect(TokenType.Keyword).value
    if dest not in Register:
        raise ParserError(f'{line}: Unexpected destination operand {dest}')
    dest_op = Register(dest)

    # parse comma
    lexer.expect(TokenType.Comma)

    # parse source
    src = lexer.expect(TokenType.Keyword, TokenType.Number)
    if src.type == TokenType.Keyword and src.value in Register:
        src_op = Register(src.value)
    elif src.type == TokenType.Number:
        src_op = int(src.value)
    else:
        raise ParserError(f'{line}: Expected source operand to be register or number')

    return Inst(line, InstType.add, [dest_op, src_op])


def parse_instructions(code: str, debug: bool = False) -> list[Inst | None]:
    lexer = Lexer(code, debug)
    line = 0
    inst: list[Inst | None] = []
    while True:
        token = lexer.next_or_none()
        if token is None:
            break

        match token.type:
            case TokenType.NewLine:
                line += 1
            case TokenType.Keyword:
                match token.value:
                    case 'mov':
                        inst.append(parse_mov(lexer, line))
                    case 'add':
                        inst.append(parse_add(lexer, line))
                    case _:
                        raise NotImplementedError(f'Keyword {token.value} is not implemented')
            case _:
                raise NotImplementedError(f'Token type {token.type} is not implemented')
    return inst
