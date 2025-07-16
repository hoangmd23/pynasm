from dataclasses import dataclass
from enum import StrEnum


class Register(StrEnum):
    rax = 'rax'
    rbx = 'rbx'
    rdx = 'rdx'
    rsi = 'rsi'
    rsp = 'rsp'
    rbp = 'rbp'
    rcx = 'rcx'
    rdi = 'rdi'
    eax = 'eax'

@dataclass
class Memory:
    base: Register | None = None
    index: Register | None = None
    scale: int = 0
    displacement: int | str = 0
