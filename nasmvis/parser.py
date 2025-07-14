from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from nasmvis.common import Register
from nasmvis.lexer import Lexer, TokenType


class ParserError(Exception):
    pass


class InstType(StrEnum):
    mov = 'mov'
    add = 'add'
    xor = 'xor'
    cmp = 'cmp'
    dec = 'dec'
    jne = 'jne'


type Operand = Register | int


@dataclass
class Inst:
    line: int
    type: InstType
    operands: list[Operand] | None = None


def parse_binop(lexer: Lexer, line: int) -> tuple[Operand, Operand]:
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

    return dest_op, src_op


def parse_mov(lexer: Lexer, line: int) -> Inst:
    return Inst(line, InstType.mov, [*parse_binop(lexer, line)])


def parse_add(lexer: Lexer, line: int) -> Inst:
    return Inst(line, InstType.add, [*parse_binop(lexer, line)])


def parse_xor(lexer: Lexer, line: int) -> Inst:
    return Inst(line, InstType.xor, [*parse_binop(lexer, line)])


def parse_cmp(lexer: Lexer, line: int) -> Inst:
    return Inst(line, InstType.cmp, [*parse_binop(lexer, line)])


def parse_dec(lexer: Lexer, line: int) -> Inst:
    dest = lexer.expect(TokenType.Keyword).value
    if dest not in Register:
        raise ParserError(f'{line}: Unexpected destination operand {dest}')
    dest_op = Register(dest)
    return Inst(line, InstType.dec, [dest_op])


def parse_jne(lexer: Lexer, line: int) -> tuple[Inst, str]:
    jmp_label = lexer.expect(TokenType.Identifier).value
    return Inst(line, InstType.jne), jmp_label


def parse_instructions(code: str, debug: bool = False) -> list[Inst | None]:
    lexer = Lexer(code, debug)
    line = 0
    inst: list[Inst | None] = []
    labels: dict[str, int] = {}
    jmp_insts: list[tuple[Inst, str]] = cast(list[tuple[Inst, str]], [])

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
                    case 'xor':
                        inst.append(parse_xor(lexer, line))
                    case 'cmp':
                        inst.append(parse_cmp(lexer, line))
                    case 'dec':
                        inst.append(parse_dec(lexer, line))
                    case 'jne':
                        jmp_inst, jmp_label = parse_jne(lexer, line)
                        jmp_insts.append((jmp_inst, jmp_label))
                        inst.append(jmp_inst)
                    case _:
                        raise NotImplementedError(f'Keyword {token.value} is not implemented')
            case TokenType.Identifier:
                # parse label
                if token.value == 'syscall':
                    pass
                else:
                    label = token.value
                    lexer.expect(TokenType.Colon)
                    while True:
                        token = lexer.peek()
                        if token.type != TokenType.NewLine:
                            break
                        line += 1
                        lexer.next()
                    labels[label] = len(inst)
            case _:
                raise NotImplementedError(f'Token type {token.type} ({token.value}) is not implemented')

    for jmp_inst, jmp_label in jmp_insts:
        if jmp_label not in labels:
            raise ParserError(f'Label {jmp_label} is not defined')
        jmp_inst.operands = [labels[jmp_label]]

    return inst
