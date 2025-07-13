from nasmvis.common import Register
from nasmvis.parser import Inst, InstType


class Machine:
    def __init__(self):
        self.inst: list[Inst | None] = []
        self.rip: int = 0
        self.registers: dict[Register, int] = {
            Register.rax: 0,
            Register.rbx: 0,
        }

    def load_inst(self, inst: list[Inst]) -> None:
        self.inst = inst
        self.reset()

    def reset(self) -> None:
        self.rip = 0

    def step(self) -> bool:
        if self.rip >= len(self.inst):
            return False

        inst = self.inst[self.rip]

        match inst:
            case Inst(line, InstType.mov, ops):
                assert len(ops) == 2
                self.rip += 1
                dest, src = ops[0], ops[1]
                match (dest, src):
                    case (Register(), int()):
                        self.registers[dest] = src
                    case _:
                        raise NotImplementedError(f'Instruction mov does not support operands {dest}, {src}')
            case Inst(line, InstType.add, ops):
                assert len(ops) == 2
                self.rip += 1
                dest, src = ops[0], ops[1]
                match (dest, src):
                    case (Register(), int()):
                        self.registers[dest] += src
                    case (Register(), Register()):
                        self.registers[dest] += self.registers[src]
                    case _:
                        raise NotImplementedError(f'Instruction add does not support operands {dest}, {src}')
            case _:
                raise NotImplementedError(f'Instruction {inst.type} is not implemented')

        return True
