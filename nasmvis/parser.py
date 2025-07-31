from dataclasses import dataclass, field
from typing import cast

from nasmvis.common import Operand, RegisterOp, MemoryOp, register_names, InstType, OperandSize, Directive, jump_inst
from nasmvis.lexer import Lexer, TokenType

type ParserResult = tuple[int | None, list[Inst | None], bytearray, dict[str, int], int]
ENTRYPOINT = 'main'


class ParserError(Exception):
    pass


@dataclass
class Inst:
    line: int
    type: InstType
    op_size: OperandSize | None = None
    operands: list[Operand] | None = field(default_factory=list)


# does not expect opening bracket, but consumes closing bracket
def parse_memory_op(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> MemoryOp:
    memory = MemoryOp()
    while True:
        token = lexer.next()
        match token.type:
            case TokenType.Keyword if token.value in register_names:
                if lexer.peek().value == '*':
                    # index
                    lexer.next()
                    if memory.index is not None:
                        raise ParserError(f'{line}: invalid effective address')
                    else:
                        if token.value in register_names:
                            memory.index = token.value
                            memory.scale = lexer.next().value
                        else:
                            raise ParserError(f'{line}: index in effective address is not a register')
                else:
                    # base or index without scale
                    # TODO: support adding same registers [rax+rax+rax+rax] = [rax*4]
                    # TODO: support arithmetic expression [2*10+3*(2+1)]
                    if memory.base is None:
                        if token.value in register_names:
                            memory.base = token.value
                        else:
                            raise ParserError(f'{line}: base in effective address is not a register')
                    elif memory.index is None:
                        if token.value in register_names:
                            memory.index = token.value
                            memory.scale = 1
                        else:
                            raise ParserError(f'{line}: index in effective address is not a register')
                    else:
                        raise ParserError(f'{line}: invalid effective address')
            case TokenType.Number | TokenType.Identifier:
                if lexer.peek().value == '*':
                    # index
                    lexer.next()
                    reg = lexer.next()
                    if reg.value in register_names:
                        memory.index = reg.value
                        memory.scale = token.value
                    else:
                        raise ParserError(f'{line}: invalid effective address')
                else:
                    # displacement
                    memory.displacement = token.value
        token = lexer.next()
        if token.value != '+':
            if token.type != TokenType.ClosingSquareBracket:
                raise ParserError(f'{line}: invalid effective address')
            break

    if memory.scale in equ_labels:
        memory.scale = int(equ_labels[memory.scale])
    elif memory.scale in data_labels:
        memory.scale = int(data_labels[memory.scale])
    else:
        memory.scale = int(memory.scale)

    if memory.displacement in equ_labels:
        memory.displacement = int(equ_labels[memory.displacement])
    elif memory.displacement in data_labels:
        memory.displacement = int(data_labels[memory.displacement])
    else:
        memory.displacement = int(memory.displacement)
    return memory


def parse_op_size(lexer: Lexer) -> OperandSize | None:
    operand_size: OperandSize | None = None
    if lexer.peek().value in OperandSize:
        operand_size = OperandSize(lexer.next().value)
    return operand_size


# TODO: we can only have 2 operand at max, remove list
# parse operands, don't consume newline
def parse_operands(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> tuple[OperandSize | None, list[Operand]]:
    if lexer.peek().type == TokenType.NewLine:
        # no operands
        return None, []

    operand_size: OperandSize | None = None
    operands: list[Operand] = []

    # parse operands
    while True:
        # if multiple operand sizes are specified, use the last one
        if operand_size is None:
            operand_size = parse_op_size(lexer)
        elif parse_op_size(lexer) is not None:
            raise ParserError(f'{line}: Size of an operand is specified two times')

        # parse operand
        token = lexer.expect(TokenType.Keyword, TokenType.Number, TokenType.Identifier, TokenType.OpeningSquareBracket)
        if token.type == TokenType.Keyword and token.value in register_names:
            operand = RegisterOp(token.value)
        elif token.type == TokenType.Number:
            operand = int(token.value)
        elif token.type == TokenType.Identifier:
            if token.value in equ_labels:
                operand = int(equ_labels[token.value])
            else:
                operand = token.value
        elif token.type == TokenType.OpeningSquareBracket:
            operand = parse_memory_op(lexer, line, data_labels, equ_labels)
        else:
            raise ParserError(f'{line}: Invalid source operand')

        operands.append(operand)

        # check if there are more operands
        token = lexer.peek()
        if token.type == TokenType.NewLine:
            break
        elif token.type == TokenType.Comma:
            lexer.expect(TokenType.Comma)
        else:
            raise ParserError(f'{line}: Expected comma or newline, but got {token}')

    return operand_size, operands


def parse_inst(inst_type: InstType, lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, inst_type, *parse_operands(lexer, line, data_labels, equ_labels))


def parse_data_and_bss(code: str, debug: bool) -> tuple[bytearray, dict[str, int], dict[str, int], int, dict[str, str], set[int]]:
    lexer = Lexer(code, debug)
    data: bytearray = bytearray()
    data_labels: dict[str, int] = {}
    bss: dict[str, int] = {} # key = name, value = offset
    bss_size: int = 0
    equ_labels: dict[str, str] = {}
    parsed_lines: set[int] = set()
    line: int = 0

    while True:
        token = lexer.next_or_none()
        if token is None:
            break

        match token.type:
            case TokenType.Identifier:
                label = token.value
                token = lexer.next()
                if token.type != TokenType.NewLine:
                    match token.value:
                        # token = lexer.next()
                        case 'db':
                            parsed_lines.add(line)
                            # define data
                            data_labels[label] = len(data)
                            while True:
                                token = lexer.next()
                                match token.type:
                                    case TokenType.String:
                                        for s in token.value:
                                            data.append(ord(s))
                                    case TokenType.Number:
                                        data.append(int(token.value) % 256)
                                    case _:
                                        raise NotImplementedError(f'Define data is not implemented for {token.type}')
                                if lexer.peek().type == TokenType.Comma:
                                    lexer.next()
                                else:
                                    break
                        case 'equ':
                            parsed_lines.add(line)
                            # TODO: currently only support numbers
                            equ_labels[label] = lexer.expect(TokenType.Number).value
                        case 'resq':
                            parsed_lines.add(line)
                            size = int(lexer.expect(TokenType.Number).value) * 8
                            bss[label] = bss_size
                            bss_size += size
                        case _:
                            # current line is not data, skip it
                            while lexer.next().type != TokenType.NewLine:
                                line += 1
                else:
                    line += 1
            case TokenType.NewLine:
                line += 1
            case _:
                # current line is not data, skip it
                while lexer.next().type != TokenType.NewLine:
                    line += 1

    return data, data_labels, bss, bss_size, equ_labels, parsed_lines


def parse_instructions(code: str, debug: bool = False) -> ParserResult:
    lexer = Lexer(code, debug)
    line = 0
    inst: list[Inst | None] = []
    labels: dict[str, int] = {}
    jmp_insts: list[tuple[Inst, str]] = cast(list[tuple[Inst, str]], [])
    data, data_labels, bss, bss_size, equ_labels, parsed_lines = parse_data_and_bss(code, debug)
    for label, offset in bss.items():
        data_labels[label] = len(data) + offset
    start_addr: int | None = None

    while True:
        token = lexer.next_or_none()
        if token is None:
            break

        # skip data and bss lines that were already processed
        if line in parsed_lines:
            while True:
                token = lexer.next_or_none()
                if token is None:
                    break
                elif token.type == TokenType.NewLine:
                    break

        # parse one line
        match token.type:
            case TokenType.NewLine:
                line += 1
                continue
            case TokenType.Keyword:
                if token.value in Directive:
                    if token.value == 'section':
                        lexer.expect(TokenType.Keyword)
                    elif token.value == 'global':
                        lexer.expect(TokenType.Identifier)
                    else:
                        raise ParserError(f'{line}: Unknown directive {token}')
                elif token.value in InstType:
                        # TODO: add support for hex numbers
                    if token.value in jump_inst:
                        jmp_inst = parse_inst(InstType(token.value), lexer, line, data_labels, equ_labels)
                        jmp_insts.append((jmp_inst, str(jmp_inst.operands[0])))
                        inst.append(jmp_inst)
                    else:
                        inst.append(parse_inst(InstType(token.value), lexer, line, data_labels, equ_labels))
                else:
                    raise NotImplementedError(f'Keyword {token.value} is not implemented')
            case TokenType.Identifier:
                # parse label
                label = token.value

                # skip colon
                if lexer.peek().type == TokenType.Colon:
                    lexer.expect(TokenType.Colon)

                labels[label] = len(inst)
                if start_addr is None and label == ENTRYPOINT:
                    start_addr = len(inst)
            case _:
                raise NotImplementedError(f'Token type {token.type} ({token.value}) is not implemented')

        if lexer.peek() is not None:
            lexer.expect(TokenType.NewLine)
            line += 1

        if debug and inst:
            print(inst[-1])

    for jmp_inst, jmp_label in jmp_insts:
        if jmp_label not in labels:
            raise ParserError(f'Label {jmp_label} is not defined')
        jmp_inst.operands = [labels[jmp_label]]

    if start_addr is None:
        start_addr = 0

    return start_addr, inst, data, data_labels, bss_size
