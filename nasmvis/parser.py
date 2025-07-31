from dataclasses import dataclass, field

from nasmvis.common import Operand, RegisterOp, MemoryOp, register_names, InstType, OperandSize, Directive, jump_inst, \
    NumberOp
from nasmvis.lexer import Lexer, TokenType

type ParserResult = tuple[int | None, list[Inst | None], bytearray, dict[str, int], int]
ENTRYPOINT = '_start'


class ParserError(Exception):
    pass


@dataclass
class Inst:
    line: int
    type: InstType
    op_size: OperandSize | None = None
    operands: list[Operand] | None = field(default_factory=list)


# does not expect opening bracket, but consumes closing bracket
def parse_memory_op(lexer: Lexer, line: int, labels: dict[str, int], data_labels: dict[str, int], equ_labels: dict[str, str]) -> MemoryOp:
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
        # TODO: add support for negative displacement
        if token.value != '+':
            if token.type != TokenType.ClosingSquareBracket:
                raise ParserError(f'{line}: invalid effective address')
            break

    memory.scale = resolve_label(memory.scale, labels, data_labels, equ_labels)
    memory.displacement = resolve_label(memory.displacement, labels, data_labels, equ_labels)

    return memory


def parse_op_size(lexer: Lexer) -> OperandSize | None:
    operand_size: OperandSize | None = None
    if lexer.peek().value in OperandSize:
        operand_size = OperandSize(lexer.next().value)
    return operand_size


def resolve_label(label: str, labels: dict[str, int], data_labels: dict[str, int], equ_labels: dict[str, str]) -> int:
    try:
        return int(label)
    except ValueError:
        res = label
        if res in equ_labels:
            res = equ_labels[res]

        if res in data_labels:
            res = data_labels[res]
        elif res in labels:
            res = labels[res]
        else:
            raise ParserError(f'Could not resolve label: {label}')

        return int(res)


# TODO: we can only have 2 operand at max, remove list
# parse operands, don't consume newline
def parse_operands(lexer: Lexer, line: int, labels: dict[str, int], data_labels: dict[str, int], equ_labels: dict[str, str]) -> tuple[OperandSize | None, list[Operand]]:
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
            operand = NumberOp(int(token.value))
        elif token.type == TokenType.Identifier:
            operand = NumberOp(resolve_label(token.value, labels, data_labels, equ_labels))
        elif token.type == TokenType.OpeningSquareBracket:
            operand = parse_memory_op(lexer, line, labels, data_labels, equ_labels)
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


def parse_inst(inst_type: InstType, lexer: Lexer, line: int, labels: dict[str, int], data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, inst_type, *parse_operands(lexer, line, labels, data_labels, equ_labels))


def parse_labels(code: str, debug: bool) -> tuple[dict[str, int], bytearray, dict[str, int], dict[str, int], int, dict[str, str], set[int]]:
    lexer = Lexer(code, debug)
    labels: dict[str, int] = {}
    inst_count: int = 0
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
            case TokenType.Keyword:
                if token.value in InstType:
                    inst_count += 1
                while lexer.next().type != TokenType.NewLine:
                    line += 1
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
                            if token.type == TokenType.Colon:
                                labels[label] = inst_count
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

    return labels, data, data_labels, bss, bss_size, equ_labels, parsed_lines


def parse_instructions(code: str, debug: bool = False) -> ParserResult:
    lexer = Lexer(code, debug)
    line = 0
    inst: list[Inst | None] = []
    labels, data, data_labels, bss, bss_size, equ_labels, parsed_lines = parse_labels(code, debug)
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
                    inst.append(parse_inst(InstType(token.value), lexer, line, labels, data_labels, equ_labels))
                else:
                    raise NotImplementedError(f'Keyword {token.value} is not implemented')
            case TokenType.Identifier:
                # parse label
                label = token.value

                # skip colon
                if lexer.peek().type == TokenType.Colon:
                    lexer.expect(TokenType.Colon)

                if start_addr is None and label == ENTRYPOINT:
                    start_addr = len(inst)
            case _:
                raise NotImplementedError(f'Token type {token.type} ({token.value}) is not implemented')

        if lexer.peek() is not None:
            lexer.expect(TokenType.NewLine)
            line += 1

        if debug and inst:
            print(inst[-1])

    if start_addr is None:
        start_addr = 0

    return start_addr, inst, data, data_labels, bss_size
