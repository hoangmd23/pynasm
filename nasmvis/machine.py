from collections import defaultdict
from enum import StrEnum

from nasmvis.common import Register, Memory
from nasmvis.parser import Inst, InstType


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


REGISTER_WIDTH = 64
MEMORY_CAPACITY = 1024

def get_sign_bit(value: int):
    return value >> (REGISTER_WIDTH - 1) & 1


class Machine:
    def __init__(self):
        self.inst: list[Inst | None] = []
        self.rip: int = 0
        self.registers: dict[Register, int] = defaultdict(int)
        self.flags: dict[Flags, bool] = {x: False for x in FlagsRegister if x is not None}
        self.reg_max_value = 2**REGISTER_WIDTH
        self.memory = bytearray(MEMORY_CAPACITY)
        self.data_labels: dict[str, int] = {}

    def load_inst_and_data(self, inst: list[Inst], data: bytearray, data_labels: dict[str, int]) -> None:
        self.inst = inst
        self.memory[:len(data)] = data
        self.data_labels = data_labels
        self.reset()

    def reset(self) -> None:
        self.rip = 0
        self.registers[Register.rbp] = len(self.memory)
        self.registers[Register.rsp] = len(self.memory)

    def set_register(self, reg: Register, value: int):
        self.registers[reg] = value % self.reg_max_value

    def compute_binop(self, dest: int, src: int, op: InstType) -> int:
        # TODO: support registers of different width, are flags set differently?
        match op:
            case InstType.add:
                res = dest + src
                res_sign = get_sign_bit(res)
                dest_sign = get_sign_bit(dest)
                src_sign = get_sign_bit(src)

                self.flags[Flags.ZF] = res == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set
                self.flags[Flags.CF] = res >= self.reg_max_value # Carry flag is set if unsigned dest + src > max value (carry)
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
            case InstType.cmp:
                res = dest - src
                res_sign = get_sign_bit(res)
                dest_sign = get_sign_bit(dest)
                src_sign = get_sign_bit(src)

                self.flags[Flags.ZF] = res == 0
                self.flags[Flags.SF] = res_sign == 1 # Sign flag is set if MSB is set
                self.flags[Flags.CF] = res < 0 # Carry flag is set if unsigned dest < src (borrow)
                # for a - b OF is set if a and b have different signs, and the result’s sign differs from the sign of a
                self.flags[Flags.OF] = (True if dest_sign != src_sign and res_sign != dest_sign else False)
                res = dest
            case _:
                raise NotImplementedError(f'Binary operator {op} is not implemented')
        return res

    def step(self) -> bool:
        if self.rip >= len(self.inst):
            return False

        inst = self.inst[self.rip]

        match inst:
            case Inst(line, op, ops) if len(ops) == 2:
                self.rip += 1
                dest, src = ops[0], ops[1]
                match (dest, src):
                    case (Register(), int()):
                        src_value = src
                    case (Register(), Register()):
                        src_value = self.registers[src]
                    case (Register(), str()):
                        src_value = self.data_labels[src]
                    case (Register(), Memory()):
                        addr = 0
                        if isinstance(src.displacement, str):
                            addr += self.data_labels[src.displacement]
                        else:
                            addr += src.displacement
                        if src.base is not None:
                            addr += self.registers[src.base]
                        if src.index is not None:
                            addr += self.registers[src.index]*src.scale
                        src_value = self.memory[addr]
                    case _:
                        raise NotImplementedError(f'Binary operator does not support operands {dest}, {src}')
                self.set_register(dest, self.compute_binop(self.registers[dest], src_value, op))
            case Inst(line, op, ops) if len(ops) == 1:
                operand = ops[0]
                match op:
                    case InstType.dec:
                        assert isinstance(operand, Register)
                        self.set_register(operand, self.registers[operand]-1)
                        self.rip += 1
                    case InstType.jne:
                        if self.flags[Flags.ZF] == 0:
                            self.rip = operand
                        else:
                            self.rip += 1
                    case InstType.push:
                        self.set_register(Register.rsp, self.registers[Register.rsp] - 8)
                        match operand:
                            case Register():
                                for i in range(8):
                                    self.memory[self.registers[Register.rsp]+i] = self.registers[operand] >> (i * 8) & 0xFF
                            case _:
                                raise NotImplementedError(f'Push is not implemented for {operand}')
                        self.rip += 1
                    case InstType.pop:
                        value = 0
                        for i in range(8):
                            value += self.memory[self.registers[Register.rsp]+i] << (i * 8)
                        self.set_register(Register.rsp, self.registers[Register.rsp] + 8)

                        match operand:
                            case Register():
                                self.set_register(operand, value)
                            case _:
                                raise NotImplementedError(f'Pop is not implemented for {operand}')
                        self.rip += 1
                    case _:
                        raise NotImplementedError(f'Unary operator {op} is not implemented')
            case _:
                raise NotImplementedError(f'Instruction {inst.type} is not implemented')

        return True
