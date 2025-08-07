from pynasm.common import OperandSize
from pynasm.cpu import get_op_size_in_bits


class Memory:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: bytearray | None = None

    def reset(self, data: bytearray) -> None:
        self.memory = bytearray(self.capacity)
        self.memory[:len(data)] = data

    def read(self, addr: int, op_size: OperandSize) -> int:
        res = 0
        for i in range(get_op_size_in_bits(op_size) // 8):
            res += self.memory[addr+i] * 256**i
        return res

    def write(self, addr: int, value: int, op_size: OperandSize):
        for i in range(get_op_size_in_bits(op_size) // 8):
            self.memory[addr + i] = value % 256
            value //= 256
