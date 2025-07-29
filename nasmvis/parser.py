from dataclasses import dataclass, field
from typing import cast


from nasmvis.common import Register, Operand, RegisterOp, MemoryOp, register_names, InstType, OperandSize
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
def parse_binop(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> tuple[OperandSize | None, list[Operand, Operand]]:
    # parse operand size if any
    operand_size: OperandSize | None = parse_op_size(lexer)

    # parse destination
    dest = lexer.expect(TokenType.Keyword, TokenType.OpeningSquareBracket)
    if dest.type == TokenType.Keyword and dest.value in register_names:
        dest_op = RegisterOp(dest.value)
    elif dest.type == TokenType.OpeningSquareBracket:
        dest_op = parse_memory_op(lexer, line, data_labels, equ_labels)
    else:
        raise ParserError(f'{line}: Invalid destination operand')

    # parse comma
    lexer.expect(TokenType.Comma)

    # parse source
    if operand_size is None:
        operand_size = parse_op_size(lexer)
    elif parse_op_size(lexer) is not None:
        raise ParserError(f'{line}: Size of an operand is specified two times')

    src = lexer.expect(TokenType.Keyword, TokenType.Number, TokenType.Identifier, TokenType.OpeningSquareBracket)

    if src.type == TokenType.Keyword and src.value in register_names:
        src_op = RegisterOp(src.value)
    elif src.type == TokenType.Number:
        src_op = int(src.value)
    elif src.type == TokenType.Identifier:
        if src.value in equ_labels:
            src_op = int(equ_labels[src.value])
        else:
            src_op = src.value
    elif src.type == TokenType.OpeningSquareBracket:
        src_op = parse_memory_op(lexer, line, data_labels, equ_labels)
    else:
        raise ParserError(f'{line}: Invalid source operand')

    return operand_size, [dest_op, src_op]


def parse_mov(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.mov, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_movzx(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.movzx, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_movsx(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.movsx, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_add(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.add, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_sub(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.sub, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_xor(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.xor, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_cmp(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.cmp, *parse_binop(lexer, line, data_labels, equ_labels))


def parse_dec(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    operand_size: OperandSize | None = parse_op_size(lexer)

    dest = lexer.expect(TokenType.Keyword, TokenType.OpeningSquareBracket)
    if dest in equ_labels:
        dest_op = equ_labels[dest.value]
    elif dest.type == TokenType.Keyword and dest.value in register_names:
        dest_op = RegisterOp(dest.value)
    elif dest.type == TokenType.OpeningSquareBracket:
        dest_op = parse_memory_op(lexer, line, data_labels, equ_labels)
    else:
        raise ParserError(f'{line}: Invalid destination operand')

    return Inst(line, InstType.dec, operand_size, [dest_op])


def parse_inc(lexer: Lexer, line: int, data_labels: dict[str, int], equ_labels: dict[str, str]) -> Inst:
    operand_size: OperandSize | None = parse_op_size(lexer)

    dest = lexer.expect(TokenType.Keyword, TokenType.OpeningSquareBracket)
    if dest in equ_labels:
        dest_op = equ_labels[dest.value]
    elif dest.type == TokenType.Keyword and dest.value in register_names:
        dest_op = RegisterOp(dest.value)
    elif dest.type == TokenType.OpeningSquareBracket:
        dest_op = parse_memory_op(lexer, line, data_labels, equ_labels)
    else:
        raise ParserError(f'{line}: Invalid destination operand')

    return Inst(line, InstType.inc, operand_size, [dest_op])


def parse_jmp(lexer: Lexer, line: int, inst_type: InstType) -> tuple[Inst, str]:
    jmp_label = lexer.expect(TokenType.Identifier).value
    return Inst(line, inst_type), jmp_label


def parse_push(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    operand_size: OperandSize | None = parse_op_size(lexer)
    value = lexer.expect(TokenType.Keyword).value
    if value in equ_labels:
        value = equ_labels[value]
    if value in register_names:
        return Inst(line, InstType.push, operand_size, [RegisterOp(value)])
    else:
        raise ParserError(f'{line}: Invalid push operand {value}')


def parse_pop(lexer: Lexer, line: int) -> Inst:
    operand_size: OperandSize | None = parse_op_size(lexer)
    token = lexer.expect(TokenType.Keyword)
    if token.value in register_names:
        return Inst(line, InstType.pop, operand_size, [RegisterOp(token.value)])
    else:
        raise ParserError(f'{line}: Invalid pop operand {token.value}')


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
                match token.value:
                    case 'section':
                        lexer.expect(TokenType.Keyword)
                    case 'global':
                        lexer.expect(TokenType.Identifier)
                    case 'mov':
                        # TODO: add support for hex numbers
                        inst.append(parse_mov(lexer, line, data_labels, equ_labels))
                    case 'movzx':
                        inst.append(parse_movzx(lexer, line, data_labels, equ_labels))
                    case 'movsx':
                        inst.append(parse_movsx(lexer, line, data_labels, equ_labels))
                    case 'add':
                        inst.append(parse_add(lexer, line, data_labels, equ_labels))
                    case 'sub':
                        inst.append(parse_sub(lexer, line, data_labels, equ_labels))
                    case 'xor':
                        inst.append(parse_xor(lexer, line, data_labels, equ_labels))
                    case 'cmp':
                        inst.append(parse_cmp(lexer, line, data_labels, equ_labels))
                    case 'inc':
                        inst.append(parse_inc(lexer, line, data_labels, equ_labels))
                    case 'dec':
                        inst.append(parse_dec(lexer, line, data_labels, equ_labels))
                    case 'jne' | 'jbe' | 'jae' | 'jmp' | 'jnz' | 'jge' | 'je':
                        jmp_inst, jmp_label = parse_jmp(lexer, line, InstType(token.value))
                        jmp_insts.append((jmp_inst, jmp_label))
                        inst.append(jmp_inst)
                    case 'push':
                        inst.append(parse_push(lexer, line, equ_labels))
                    case 'pop':
                        inst.append(parse_pop(lexer, line))
                    case 'call':
                        jmp_inst, jmp_label = parse_jmp(lexer, line, InstType.call)
                        jmp_insts.append((jmp_inst, jmp_label))
                        inst.append(jmp_inst)
                    case 'ret':
                        inst.append(Inst(line, InstType.ret))
                    case 'exit':
                        inst.append(Inst(line, InstType.exit))
                    case _:
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
