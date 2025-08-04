from enum import StrEnum

from pynasm.common import Register, Registers, R64, RL, R32, R16, RH, RegisterOp, MemoryOp, OperandSize, NumberOp, \
    Operand, jump_inst
from pynasm.parser import Inst, InstType


type RegisterType = R64 | R32 | R16 | RH | RL


class MachineException(Exception):
    pass


class Flags(StrEnum):
    CF = 'CF'
    ZF = 'ZF'
    SF = 'SF'
    OF = 'OF'


FlagsRegister = [
    Flags.CF,
    None,
    None,
    None,
    None,
    None,
    Flags.ZF,
    Flags.SF,
    None,
    None,
    None,
    Flags.OF,
]


MEMORY_CAPACITY = 1024
R64_WIDTH = 64
R64_MAX_VALUE = 2 ** R64_WIDTH
R32_WIDTH = 32
R32_MAX_VALUE = 2 ** R32_WIDTH
R16_WIDTH = 16
R16_MAX_VALUE = 2 ** R16_WIDTH
R8_WIDTH = 8
R8_MAX_VALUE = 2 ** R8_WIDTH


def get_sign_bit(value: int, reg_width: int):
    return value >> (reg_width - 1) & 1


def get_reg_by_name(name: str) -> RegisterType:
    if name in R64:
        reg = R64(name)
    elif name in R32:
        reg = R32(name)
    elif name in R16:
        reg = R16(name)
    elif name in RH:
        reg = RH(name)
    elif name in RL:
        reg = RL(name)
    else:
        raise MachineException(f'Unknown register: {name}')
    return reg


def get_reg_op_size(name: str) -> OperandSize:
    if name in R64:
        op_size = OperandSize.qword
    elif name in R32:
        op_size = OperandSize.dword
    elif name in R16:
        op_size = OperandSize.word
    elif name in RH:
        op_size = OperandSize.byte
    elif name in RL:
        op_size = OperandSize.byte
    else:
        raise MachineException(f'Unknown register: {name}')
    return op_size


def get_op_bit_count(op_size: OperandSize):
    match op_size:
        case OperandSize.byte:
            return R8_WIDTH
        case OperandSize.word:
            return R16_WIDTH
        case OperandSize.dword:
            return R32_WIDTH
        case OperandSize.qword:
            return R64_WIDTH
        case _:
            assert False


def get_op_max_value(op: OperandSize):
    match op:
        case OperandSize.byte:
            return R8_MAX_VALUE
        case OperandSize.word:
            return R16_MAX_VALUE
        case OperandSize.dword:
            return R32_MAX_VALUE
        case OperandSize.qword:
            return R64_MAX_VALUE
        case _:
            assert False


def get_inst_operands_sizes(dest: Operand, src: Operand, op_size: OperandSize | None) -> tuple[OperandSize, OperandSize]:
    if isinstance(dest, RegisterOp):
        dest_op_size = get_reg_op_size(dest.value)
    else:
        dest_op_size = op_size

    if isinstance(src, RegisterOp):
        src_op_size = get_reg_op_size(src.value)
    else:
        src_op_size = op_size

    assert dest_op_size is not None or src_op_size is not None

    if dest_op_size is None:
        dest_op_size = src_op_size
    if src_op_size is None:
        src_op_size = dest_op_size

    return dest_op_size, src_op_size


class Machine:
    def __init__(self):
        self.inst: list[Inst | None] = []
        self.rip: int = 0

        self.reg64: dict[R64, Register] = { r.r64: r for r in Registers }
        self.reg32: dict[R32, Register] = { r.r32: r for r in Registers }
        self.reg16: dict[R16, Register] = { r.r16: r for r in Registers }
        self.reg8h: dict[RH, Register] = { r.rh: r for r in Registers if r.rh is not None }
        self.reg8l: dict[RL, Register] = { r.rl: r for r in Registers }

        self.flags: dict[Flags, bool] = {x: False for x in FlagsRegister if x is not None}
        self.memory = bytearray(MEMORY_CAPACITY)
        self.data_labels: dict[str, int] = {}
        self.entrypoint: int = -1
        self.running: bool = False

    def load_inst_and_data(self, entrypoint: int, inst: list[Inst], data: bytearray, data_labels: dict[str, int], bss_size: int) -> None:
        self.inst = inst
        self.memory = bytearray(MEMORY_CAPACITY)
        self.memory[:len(data)] = data
        self.data_labels = data_labels
        self.entrypoint = entrypoint
        self.reset()

    def reset(self) -> None:
        self.rip = self.entrypoint
        for r in self.reg64:
            self.set_register(r, 0, False)
        self.set_flags(False)
        self.set_register(R64.rbp, len(self.memory), False)
        self.set_register(R64.rsp, len(self.memory), False)
        self.running = True

    def set_flags(self, value: bool):
        for flag in self.flags:
            self.flags[flag] = value

    def set_register(self, reg: R64 | R32 | R16 | RH | RL | str, value: int, clear_upper_bits: bool):
        if isinstance(reg, str):
            reg = get_reg_by_name(reg)
        match reg:
            case R64():
                self.reg64[reg].value = value % R64_MAX_VALUE
            case R32():
                if clear_upper_bits:
                    self.reg32[reg].value = 0
                self.reg32[reg].value = (self.reg32[reg].value & 0xFFFFFFFF00000000) | (value % R32_MAX_VALUE)
            case R16():
                self.reg16[reg].value = (self.reg16[reg].value & 0xFFFFFFFFFFFF0000) | (value % R16_MAX_VALUE)
            case RH():
                self.reg8h[reg].value = (self.reg8h[reg].value & 0xFFFFFFFFFFFF00FF) | ((value % R8_MAX_VALUE) << 8)
            case RL():
                self.reg8l[reg].value = (self.reg8l[reg].value & 0xFFFFFFFFFFFFFF00) | (value % R8_MAX_VALUE)
            case _:
                raise NotImplementedError(f'{reg.__class__} is not implemented for set_register')

    def get_register(self, reg: R64 | R32 | R16 | RH | RL | str) -> int:
        if isinstance(reg, str):
            reg = get_reg_by_name(reg)
        match reg:
            case R64():
                return self.reg64[reg].value
            case R32():
                return self.reg32[reg].value & 0xFFFFFFFF
            case R16():
                return self.reg16[reg].value & 0xFFFF
            case RH():
                return (self.reg8h[reg].value & 0xFFFF) >> 8
            case RL():
                return self.reg8l[reg].value & 0xFF
            case _:
                raise NotImplementedError(f'{reg.__class__} is not implemented for get_register')

    def read_memory(self, addr: int, op_size: OperandSize) -> int:
        res = 0
        for i in range(get_op_bit_count(op_size) // 8):
            res += self.memory[addr+i] * 256**i
        return res

    def write_memory(self, addr: int, value: int, op_size: OperandSize):
        for i in range(get_op_bit_count(op_size) // 8):
            self.memory[addr + i] = value % 256
            value //= 256

    def read_flag(self, flag: Flags) -> bool:
        return self.flags[flag]

    def compute_binop(self, dest: int, src: int, op: InstType, dest_op_size: OperandSize, src_op_size: OperandSize) -> tuple[int, bool]:
        # some operations on a 32-bit register clear the upper 32 bits of the corresponding 64-bit register
        match op:
            case InstType.add | InstType.inc:
                assert dest_op_size == src_op_size
                reg_width = get_op_bit_count(dest_op_size)
                max_value = get_op_max_value(dest_op_size)
                res = dest + src
                res_sign = get_sign_bit(res, reg_width)
                dest_sign = get_sign_bit(dest, reg_width)
                src_sign = get_sign_bit(src, reg_width)

                self.flags[Flags.ZF] = res % max_value == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set

                if op == InstType.add:
                    self.flags[Flags.CF] = res >= max_value # Carry flag is set if unsigned dest + src > max value (carry)
                # for a + b OF is set if a and b have the same sign, but the result has a different sign
                self.flags[Flags.OF] = (True if dest_sign == src_sign and res_sign != dest_sign else False)
                clear_upper_bits = True
            case InstType.xor:
                assert dest_op_size == src_op_size
                reg_width = get_op_bit_count(dest_op_size)
                max_value = get_op_max_value(dest_op_size)
                res = dest ^ src
                self.flags[Flags.ZF] = res % max_value == 0
                self.flags[Flags.SF] = get_sign_bit(res, reg_width) == 1
                self.flags[Flags.CF] = False
                self.flags[Flags.OF] = False
                clear_upper_bits = True
            case InstType.mov | InstType.movzx:
                res = src
                clear_upper_bits = True
            case InstType.movsx:
                reg_width = get_op_bit_count(dest_op_size)
                # TODO: currently only supports 1 byte source
                sign = get_sign_bit(src, 8)
                res = src
                if sign == 1:
                    for i in range(reg_width - 8):
                        res |= 1 << (i + 8)
                clear_upper_bits = True
            case InstType.cmp | InstType.sub | InstType.dec:
                assert dest_op_size == src_op_size
                reg_width = get_op_bit_count(dest_op_size)
                max_value = get_op_max_value(dest_op_size)
                res = dest - src
                res_sign = get_sign_bit(res, reg_width)
                dest_sign = get_sign_bit(dest, reg_width)
                src_sign = get_sign_bit(src, reg_width)

                self.flags[Flags.ZF] = res % max_value == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set

                if op != InstType.dec:
                    self.flags[Flags.CF] = res < 0 # Carry flag is set if unsigned dest < src (borrow)
                # for a - b OF is set if a and b have different signs, and the result’s sign differs from the sign of a
                self.flags[Flags.OF] = (True if dest_sign != src_sign and res_sign != dest_sign else False)
                if op == InstType.cmp:
                    res = dest
                clear_upper_bits = True
            case _:
                raise NotImplementedError(f'Binary operator {op} is not implemented')
        return res, clear_upper_bits

    def push_onto_stack(self, value: int, op_size: OperandSize) -> None:
        byte_count = get_op_bit_count(op_size) // 8
        self.set_register(R64.rsp, self.get_register(R64.rsp) - byte_count, clear_upper_bits=False)
        for i in range(byte_count):
            self.memory[self.get_register(R64.rsp) + i] = value >> (i * 8) & 0xFF

    def pop_from_stack(self, op_size: OperandSize) -> int:
        if self.get_register(R64.rsp) >= len(self.memory):
            raise MachineException(f'Stack underflow')
        byte_count = get_op_bit_count(op_size) // 8
        value = self.read_memory(self.get_register(R64.rsp), op_size)
        self.set_register(R64.rsp, self.get_register(R64.rsp) + byte_count, clear_upper_bits=False)
        return value

    def calc_effective_addr(self, op: MemoryOp) -> int:
        res = op.displacement
        if op.base is not None:
            res += self.get_register(op.base)
        if op.index is not None:
            res += self.get_register(op.index) * op.scale
        return res

    def get_operand_value(self, inst: InstType, op: Operand, op_size: OperandSize) -> int:
        match op:
            case RegisterOp(value):
                res = self.get_register(value)
            case NumberOp(value):
                res = value
                if inst in [InstType.add, InstType.sub, InstType.cmp, InstType.xor]:
                    # max supported immediate value for add is 32-bit value
                    if op_size == OperandSize.qword:
                        res %= R32_MAX_VALUE
                        # sign extend
                        if get_sign_bit(res, R32_WIDTH) == 1:
                            for j in range(R64_WIDTH - 32):
                                res |= 1 << (j + 32)
                    else:
                        res %= get_op_max_value(op_size)
            case MemoryOp() as mem_op:
                res = self.read_memory(self.calc_effective_addr(mem_op), op_size)
            case _:
                raise NotImplementedError(f'Unknown operand type {op}\n')
        return res

    def should_jump(self, inst: InstType) -> bool:
        match inst:
            case InstType.jne:
                return not self.flags[Flags.ZF]
            case InstType.je:
                return self.flags[Flags.ZF]
            case InstType.jbe:
                return self.flags[Flags.CF] or self.flags[Flags.ZF]
            case InstType.jae:
                return not self.flags[Flags.CF]
            case InstType.jnz:
                return not self.flags[Flags.ZF]
            case InstType.jge:
                return self.flags[Flags.SF] == self.flags[Flags.OF]
            case InstType.jmp:
                return True
            case _:
                raise NotImplementedError(f'Jump instruction {inst} is not implemented')

    def step(self) -> bool:
        if not self.running:
            return False

        inst = self.inst[self.rip]

        match inst:
            case Inst(line, op, op_size, ops) if len(ops) == 2 or op == InstType.inc or op == InstType.dec:
                self.rip += 1
                if op == InstType.inc or op == InstType.dec:
                    dest = ops[0]
                    src = NumberOp(1)
                else:
                    dest, src = ops[0], ops[1]
                dest_op_size, src_op_size = get_inst_operands_sizes(dest, src, op_size)
                dest_value = self.get_operand_value(op, dest, dest_op_size)
                src_value = self.get_operand_value(op, src, src_op_size)
                res, clear_upper_bits = self.compute_binop(dest_value, src_value, op, dest_op_size, src_op_size)

                if isinstance(dest, RegisterOp):
                    self.set_register(dest.value, res, clear_upper_bits)
                else:
                    self.write_memory(self.calc_effective_addr(dest), res, dest_op_size)
            case Inst(line, op, op_size, ops) if len(ops) == 1:
                operand = ops[0]
                match op:
                    case x if x in jump_inst:
                        if self.should_jump(x):
                            self.rip = operand.value
                        else:
                            self.rip += 1
                    case InstType.push:
                        match operand:
                            case RegisterOp():
                                self.push_onto_stack(self.get_register(operand.value), get_reg_op_size(operand.value))
                            case NumberOp():
                                self.push_onto_stack(operand.value, OperandSize.qword)
                            case MemoryOp():
                                assert op_size is not None
                                value = self.read_memory(self.calc_effective_addr(operand), op_size)
                                self.push_onto_stack(value, op_size)
                            case _:
                                raise NotImplementedError(f'{line}: Push is not implemented for {operand}')
                        self.rip += 1
                    case InstType.pop:
                        match operand:
                            case RegisterOp():
                                value_on_stack = self.pop_from_stack(get_reg_op_size(operand.value))
                                self.set_register(operand.value, value_on_stack, clear_upper_bits=False)
                            case MemoryOp():
                                assert op_size is not None
                                value_on_stack = self.pop_from_stack(op_size)
                                self.write_memory(self.calc_effective_addr(operand), value_on_stack, op_size)
                            case _:
                                raise NotImplementedError(f'{line}: Pop is not implemented for {operand}')
                        self.rip += 1
                    case InstType.call:
                        self.push_onto_stack(self.rip + 1, OperandSize.qword)
                        self.rip = operand.value
                    case _:
                        raise NotImplementedError(f'{line}: Unary operator {op} is not implemented')
            case Inst(_, InstType.ret, _, _):
                self.rip = self.pop_from_stack(OperandSize.qword)
            case Inst(_, InstType.exit, _, _):
                self.running = False
            case _:
                raise NotImplementedError(f'Instruction {inst.type} is not implemented')

        if self.rip >= len(self.inst):
            self.running = False

        return True
