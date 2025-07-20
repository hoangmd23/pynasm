from enum import StrEnum

from nasmvis.common import Register, Registers, R64, RL, R32, R16, RH, RegisterOp, MemoryOp
from nasmvis.parser import Inst, InstType


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


def get_reg_max_value(reg: RegisterType) -> int:
    match reg.name:
        case x if x in R64:
            return R64_MAX_VALUE
        case x if x in R32:
            return R32_MAX_VALUE
        case x if x in R16:
            return R16_MAX_VALUE
        case x if x in RH or x in RL:
            return R8_MAX_VALUE
        case _:
            raise MachineException(f'Unknown register: {reg}')


def get_reg_width(reg: RegisterType) -> int:
    match reg.name:
        case x if x in R64:
            return R64_WIDTH
        case x if x in R32:
            return R32_WIDTH
        case x if x in R16:
            return R16_WIDTH
        case x if x in RH or x in RL:
            return R8_WIDTH
        case _:
            raise MachineException(f'Unknown register: {reg}')


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

    def load_inst_and_data(self, entrypoint: int, inst: list[Inst], data: bytearray, data_labels: dict[str, int]) -> None:
        self.inst = inst
        self.memory[:len(data)] = data
        self.data_labels = data_labels
        self.entrypoint = entrypoint
        self.reset()

    def reset(self) -> None:
        self.rip = self.entrypoint
        self.set_register(R64.rbp, len(self.memory))
        self.set_register(R64.rsp, len(self.memory))
        self.running = True

    def set_register(self, reg: R64 | R32 | R16 | RH | RL | str, value: int):
        if isinstance(reg, str):
            reg = get_reg_by_name(reg)
        match reg:
            case R64():
                self.reg64[reg].value = value % R64_MAX_VALUE
            case R32():
                self.reg32[reg].value = value % R32_MAX_VALUE
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
            case _:
                raise NotImplementedError(f'{reg.__class__} is not implemented for get_register')


    def compute_binop(self, dest: int, src: int, op: InstType, max_value: int, reg_width: int) -> int:
        # TODO: support registers of different width, are flags set differently?
        match op:
            case InstType.add:
                res = dest + src
                res_sign = get_sign_bit(res, reg_width)
                dest_sign = get_sign_bit(dest, reg_width)
                src_sign = get_sign_bit(src, reg_width)

                self.flags[Flags.ZF] = res % max_value == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set
                self.flags[Flags.CF] = res >= max_value # Carry flag is set if unsigned dest + src > max value (carry)
                # for a + b OF is set if a and b have the same sign, but the result has a different sign
                self.flags[Flags.OF] = (True if dest_sign == src_sign and res_sign != dest_sign else False)
            case InstType.xor:
                self.flags[Flags.ZF] = True
                self.flags[Flags.SF] = False
                self.flags[Flags.CF] = False
                self.flags[Flags.OF] = False
                res = dest ^ src
            case InstType.mov:
                res = src
            case InstType.cmp | InstType.sub:
                res = dest - src
                res_sign = get_sign_bit(res, reg_width)
                dest_sign = get_sign_bit(dest, reg_width)
                src_sign = get_sign_bit(src, reg_width)

                self.flags[Flags.ZF] = res % max_value == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set
                self.flags[Flags.CF] = res < 0 # Carry flag is set if unsigned dest < src (borrow)
                # for a - b OF is set if a and b have different signs, and the result’s sign differs from the sign of a
                self.flags[Flags.OF] = (True if dest_sign != src_sign and res_sign != dest_sign else False)
                if op == InstType.cmp:
                    res = dest
            case _:
                raise NotImplementedError(f'Binary operator {op} is not implemented')
        return res

    def push_onto_stack(self, value: int) -> None:
        self.set_register(R64.rsp, self.get_register(R64.rsp) - 8)
        for i in range(8):
            self.memory[self.get_register(R64.rsp) + i] = value >> (i * 8) & 0xFF

    def pop_from_stack(self) -> int:
        if self.get_register(R64.rsp) >= len(self.memory):
            raise MachineException(f'Stack underflow')
        value = 0
        for i in range(8):
            value += self.memory[self.get_register(R64.rsp) + i] << (i * 8)
        self.set_register(R64.rsp, self.get_register(R64.rsp) + 8)
        return value

    def step(self) -> bool:
        if not self.running:
            return False

        inst = self.inst[self.rip]

        match inst:
            case Inst(line, op, ops) if len(ops) == 2:
                self.rip += 1
                dest, src = ops[0], ops[1]
                match (dest, src):
                    case (RegisterOp(), int()):
                        src_value = src
                        reg_max_value = get_reg_max_value(dest)
                        reg_width = get_reg_width(dest)
                    case (RegisterOp(), RegisterOp()):
                        src_value = self.get_register(src.name)
                        reg_max_value = get_reg_max_value(dest)
                        reg_width = get_reg_width(dest)
                    case (RegisterOp(), str()):
                        src_value = self.data_labels[src]
                        reg_max_value = get_reg_max_value(dest)
                        reg_width = get_reg_width(dest)
                    case (RegisterOp(), MemoryOp()):
                        addr = 0
                        if isinstance(src.displacement, str):
                            addr += self.data_labels[src.displacement]
                        else:
                            addr += src.displacement
                        if src.base is not None:
                            addr += self.get_register(src.base)
                        if src.index is not None:
                            addr += self.get_register(src.index)*src.scale
                        src_value = self.memory[addr]
                        reg_max_value = get_reg_max_value(dest)
                        reg_width = get_reg_width(dest)
                    case _:
                        raise NotImplementedError(f'{line}: Binary operator does not support operands {dest}, {src}')
                self.set_register(dest.name, self.compute_binop(self.get_register(dest.name), src_value, op, reg_max_value, reg_width))
            case Inst(line, op, ops) if len(ops) == 1:
                operand = ops[0]
                match op:
                    case InstType.dec:
                        assert isinstance(operand, RegisterOp)
                        self.set_register(operand.name, self.get_register(operand.name)-1)
                        self.rip += 1
                    case InstType.jne:
                        if self.flags[Flags.ZF] == 0:
                            self.rip = operand
                        else:
                            self.rip += 1
                    case InstType.push:
                        match operand:
                            case Register():
                                self.push_onto_stack(self.get_register(operand.name))
                            case _:
                                raise NotImplementedError(f'{line}: Push is not implemented for {operand}')
                        self.rip += 1
                    case InstType.pop:
                        value = self.pop_from_stack()
                        match operand:
                            case Register():
                                self.set_register(operand, value)
                            case _:
                                raise NotImplementedError(f'{line}: Pop is not implemented for {operand}')
                        self.rip += 1
                    case InstType.call:
                        self.push_onto_stack(self.rip + 1)
                        self.rip = operand
                    case _:
                        raise NotImplementedError(f'{line}: Unary operator {op} is not implemented')
            case Inst(_, InstType.ret, _):
                self.rip = self.pop_from_stack()
            case Inst(_, InstType.exit, _):
                self.running = False
            case _:
                raise NotImplementedError(f'Instruction {inst.type} is not implemented')

        if self.rip >= len(self.inst):
            self.running = False

        return True
