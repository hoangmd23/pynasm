from pynasm.common import R64, RegisterOp, MemoryOp, OperandSize, NumberOp, \
    Operand, jump_inst, OPERAND_SIZE_MAX_VALUE, \
    OPERAND_SIZE_IN_BITS
from pynasm.cpu import Cpu, get_sign_bit, get_op_size_max_value, get_inst_operands_sizes, \
    get_op_size_by_reg_name, get_op_size_in_bytes
from pynasm.memory import Memory
from pynasm.parser import Inst, InstType


class MachineException(Exception):
    pass


MEMORY_CAPACITY = 1024


class Machine:
    def __init__(self):
        self.inst: list[Inst | None] = []
        self.memory: Memory = Memory(MEMORY_CAPACITY)
        self.cpu = Cpu()
        self.entrypoint: int = -1
        self.running: bool = False
        self.data: bytearray | None = None

    def load_inst_and_data(self, entrypoint: int, inst: list[Inst], data: bytearray, data_labels: dict[str, int], bss_size: int) -> None:
        self.inst = inst
        self.data = data
        self.entrypoint = entrypoint
        self.reset()

    def reset(self) -> None:
        self.cpu.reset(self.entrypoint, self.memory.capacity)
        self.memory.reset(self.data)
        self.running = True

    def push_onto_stack(self, value: int, op_size: OperandSize) -> None:
        byte_count = get_op_size_in_bytes(op_size)
        self.cpu.set_register(R64.rsp, self.cpu.get_register(R64.rsp) - byte_count, clear_upper_bits=False)
        self.memory.write(self.cpu.get_register(R64.rsp), value, op_size)

    def pop_from_stack(self, op_size: OperandSize) -> int:
        if self.cpu.get_register(R64.rsp) >= self.memory.capacity:
            raise MachineException(f'Stack underflow')
        byte_count = get_op_size_in_bytes(op_size)
        value = self.memory.read(self.cpu.get_register(R64.rsp), op_size)
        self.cpu.set_register(R64.rsp, self.cpu.get_register(R64.rsp) + byte_count, clear_upper_bits=False)
        return value

    def calc_effective_addr(self, op: MemoryOp) -> int:
        res = op.displacement
        if op.base is not None:
            res += self.cpu.get_register(op.base)
        if op.index is not None:
            res += self.cpu.get_register(op.index) * op.scale
        return res

    def get_operand_value(self, inst: InstType, op: Operand, op_size: OperandSize) -> int:
        match op:
            case RegisterOp(value):
                res = self.cpu.get_register(value)
            case NumberOp(value):
                res = value
                if inst in [InstType.add, InstType.sub, InstType.cmp, InstType.xor]:
                    # max supported immediate value for add is 32-bit value
                    if op_size == OperandSize.qword:
                        res %= OPERAND_SIZE_MAX_VALUE[OperandSize.dword]
                        # sign extend
                        if get_sign_bit(res, OPERAND_SIZE_IN_BITS[OperandSize.dword]) == 1:
                            for j in range(OPERAND_SIZE_IN_BITS[OperandSize.qword] - 32):
                                res |= 1 << (j + 32)
                    else:
                        res %= get_op_size_max_value(op_size)
            case MemoryOp() as mem_op:
                res = self.memory.read(self.calc_effective_addr(mem_op), op_size)
            case _:
                raise NotImplementedError(f'Unknown operand type {op}\n')
        return res

    def step(self) -> bool:
        if not self.running:
            return False

        inst = self.inst[self.cpu.rip]

        match inst:
            case Inst(line, op, op_size, ops) if len(ops) == 2 or op == InstType.inc or op == InstType.dec:
                self.cpu.rip += 1
                if op == InstType.inc or op == InstType.dec:
                    dest = ops[0]
                    src = NumberOp(1)
                else:
                    dest, src = ops[0], ops[1]
                dest_op_size, src_op_size = get_inst_operands_sizes(dest, src, op_size)
                dest_value = self.get_operand_value(op, dest, dest_op_size)
                src_value = self.get_operand_value(op, src, src_op_size)
                res, clear_upper_bits = self.cpu.compute_binop(dest_value, src_value, op, dest_op_size, src_op_size)

                if isinstance(dest, RegisterOp):
                    self.cpu.set_register(dest.value, res, clear_upper_bits)
                else:
                    self.memory.write(self.calc_effective_addr(dest), res, dest_op_size)
            case Inst(line, op, op_size, ops) if len(ops) == 1:
                operand = ops[0]
                match op:
                    case x if x in jump_inst:
                        if self.cpu.should_jump(x):
                            self.cpu.rip = operand.value
                        else:
                            self.cpu.rip += 1
                    case InstType.push:
                        match operand:
                            case RegisterOp():
                                self.push_onto_stack(self.cpu.get_register(operand.value), get_op_size_by_reg_name(operand.value))
                            case NumberOp():
                                self.push_onto_stack(operand.value, OperandSize.qword)
                            case MemoryOp():
                                assert op_size is not None
                                value = self.memory.read(self.calc_effective_addr(operand), op_size)
                                self.push_onto_stack(value, op_size)
                            case _:
                                raise NotImplementedError(f'{line}: Push is not implemented for {operand}')
                        self.cpu.rip += 1
                    case InstType.pop:
                        match operand:
                            case RegisterOp():
                                value_on_stack = self.pop_from_stack(get_op_size_by_reg_name(operand.value))
                                self.cpu.set_register(operand.value, value_on_stack, clear_upper_bits=False)
                            case MemoryOp():
                                assert op_size is not None
                                value_on_stack = self.pop_from_stack(op_size)
                                self.memory.write(self.calc_effective_addr(operand), value_on_stack, op_size)
                            case _:
                                raise NotImplementedError(f'{line}: Pop is not implemented for {operand}')
                        self.cpu.rip += 1
                    case InstType.call:
                        self.push_onto_stack(self.cpu.rip + 1, OperandSize.qword)
                        self.cpu.rip = operand.value
                    case _:
                        raise NotImplementedError(f'{line}: Unary operator {op} is not implemented')
            case Inst(_, InstType.ret, _, _):
                self.cpu.rip = self.pop_from_stack(OperandSize.qword)
            case Inst(_, InstType.exit, _, _):
                self.running = False
            case _:
                raise NotImplementedError(f'Instruction {inst.type} is not implemented')

        if self.cpu.rip >= len(self.inst):
            self.running = False

        return True
