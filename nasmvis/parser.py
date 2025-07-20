from dataclasses import dataclass, field
from typing import cast


from nasmvis.common import Register, Operand, RegisterOp, MemoryOp, register_names, InstType
from nasmvis.lexer import Lexer, TokenType


type ParserResult = tuple[int | None, list[Inst | None], bytearray, dict[str, int]]
ENTRYPOINT = 'main'


class ParserError(Exception):
    pass


@dataclass
class Inst:
    line: int
    type: InstType
    operands: list[Operand] | None = field(default_factory=list)


def parse_binop(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> tuple[Operand, Operand]:
    # parse destination
    dest = lexer.expect(TokenType.Keyword).value
    if dest in equ_labels:
        dest = equ_labels[dest]
    if dest not in register_names:
        raise ParserError(f'{line}: Unexpected destination operand {dest}')
    dest_op = RegisterOp(dest)

    # parse comma
    lexer.expect(TokenType.Comma)

    # parse source
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
                                memory.scale = int(lexer.next().value)
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
                        if reg.value in Register:
                            memory.index = reg.value
                            memory.scale = int(token.value)
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

        src_op = memory
    else:
        raise ParserError(f'{line}: Expected source operand to be register or number')

    return dest_op, src_op


def parse_mov(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.mov, [*parse_binop(lexer, line, equ_labels)])


def parse_add(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.add, [*parse_binop(lexer, line, equ_labels)])


def parse_sub(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.sub, [*parse_binop(lexer, line, equ_labels)])


def parse_xor(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.xor, [*parse_binop(lexer, line, equ_labels)])


def parse_cmp(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    return Inst(line, InstType.cmp, [*parse_binop(lexer, line, equ_labels)])


def parse_dec(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    dest = lexer.expect(TokenType.Keyword).value
    if dest in equ_labels:
        dest = equ_labels[dest]
    if dest not in register_names:
        raise ParserError(f'{line}: Unexpected destination operand {dest}')
    dest_op = RegisterOp(dest)
    return Inst(line, InstType.dec, [dest_op])


def parse_jmp(lexer: Lexer, line: int, inst_type: InstType) -> tuple[Inst, str]:
    jmp_label = lexer.expect(TokenType.Identifier).value
    return Inst(line, inst_type), jmp_label


def parse_push(lexer: Lexer, line: int, equ_labels: dict[str, str]) -> Inst:
    value = lexer.expect(TokenType.Keyword).value
    if value in equ_labels:
        value = equ_labels[value]
    if value in register_names:
        return Inst(line, InstType.push, [RegisterOp(value)])
    else:
        raise ParserError(f'{line}: Invalid push operand {value}')


def parse_pop(lexer: Lexer, line: int) -> Inst:
    token = lexer.expect(TokenType.Keyword)
    if token.value in register_names:
        return Inst(line, InstType.pop, [RegisterOp(token.value)])
    else:
        raise ParserError(f'{line}: Invalid pop operand {token.value}')


def parse_instructions(code: str, debug: bool = False) -> ParserResult:
    lexer = Lexer(code, debug)
    line = 0
    inst: list[Inst | None] = []
    labels: dict[str, int] = {}
    jmp_insts: list[tuple[Inst, str]] = cast(list[tuple[Inst, str]], [])
    data: bytearray = bytearray()
    data_labels: dict[str, int] = {}
    equ_labels: dict[str, str] = {}
    start_addr: int | None = None

    while True:
        token = lexer.next_or_none()
        if token is None:
            break

        match token.type:
            case TokenType.NewLine:
                line += 1
            case TokenType.Keyword:
                match token.value:
                    case 'section':
                        lexer.expect(TokenType.Dot)
                        lexer.expect(TokenType.Keyword)
                    case 'mov':
                        # TODO: add support for hex numbers
                        inst.append(parse_mov(lexer, line, equ_labels))
                    case 'add':
                        inst.append(parse_add(lexer, line, equ_labels))
                    case 'sub':
                        inst.append(parse_sub(lexer, line, equ_labels))
                    case 'xor':
                        inst.append(parse_xor(lexer, line, equ_labels))
                    case 'cmp':
                        inst.append(parse_cmp(lexer, line, equ_labels))
                    case 'dec':
                        inst.append(parse_dec(lexer, line, equ_labels))
                    case 'jne':
                        jmp_inst, jmp_label = parse_jmp(lexer, line, InstType.jne)
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
                if token.value == 'syscall':
                    pass
                else:
                    # parse label
                    label = token.value

                    # skip colon
                    if lexer.peek().type == TokenType.Colon:
                        lexer.expect(TokenType.Colon)

                    match lexer.peek().value:
                        case 'db':
                            # define data
                            data_labels[label] = len(data)
                            token = lexer.next()
                            match token.value:
                                case 'db':
                                    while True:
                                        token = lexer.next()
                                        match token.type:
                                            case TokenType.String:
                                                for s in token.value:
                                                    data.append(ord(s))
                                            case TokenType.Number:
                                                data.append(int(token.value))
                                            case _:
                                                raise NotImplementedError(f'Define data is not implemented for {token.type}')
                                        if lexer.peek().type == TokenType.Comma:
                                            lexer.next()
                                        else:
                                            break
                                case _:
                                    raise NotImplementedError(f'{token.value} is not implemented')
                        case 'equ':
                            # TODO: currently only support numbers
                            lexer.expect(TokenType.Keyword) # consume equ
                            equ_labels[label] = lexer.expect(TokenType.Number).value
                        case _:
                            # define label
                            while True:
                                token = lexer.peek()
                                if token.type != TokenType.NewLine:
                                    break
                                line += 1
                                lexer.next()
                            labels[label] = len(inst)
                            if start_addr is None and label == ENTRYPOINT:
                                start_addr = len(inst)
            case _:
                raise NotImplementedError(f'Token type {token.type} ({token.value}) is not implemented')

    for jmp_inst, jmp_label in jmp_insts:
        if jmp_label not in labels:
            raise ParserError(f'Label {jmp_label} is not defined')
        jmp_inst.operands = [labels[jmp_label]]

    return start_addr, inst, data, data_labels
