from dataclasses import dataclass
from enum import StrEnum


class Directive(StrEnum):
    section = 'section'
    global_ = 'global'


class InstType(StrEnum):
    mov = 'mov'
    movzx = 'movzx'
    movsx = 'movsx'
    add = 'add'
    sub = 'sub'
    xor = 'xor'
    cmp = 'cmp'
    inc = 'inc'
    dec = 'dec'
    jmp = 'jmp'
    je = 'je'
    jne = 'jne'
    jnz = 'jnz'
    jbe = 'jbe'
    jae = 'jae'
    jge = 'jge'
    call = 'call'
    ret = 'ret'
    exit = 'exit'
    push = 'push'
    pop = 'pop'


jump_inst = ['jne', 'jbe', 'jae', 'jmp', 'jnz', 'jge', 'je', 'call']


class R64(StrEnum):
    rax = 'rax'
    rbx = 'rbx'
    rcx = 'rcx'
    rdx = 'rdx'
    rsi = 'rsi'
    rdi = 'rdi'
    rbp = 'rbp'
    rsp = 'rsp'
    r8 = 'r8'
    r9 = 'r9'
    r10 = 'r10'
    r11 = 'r11'
    r12 = 'r12'
    r13 = 'r13'
    r14 = 'r14'
    r15 = 'r15'


class R32(StrEnum):
    eax = 'eax'
    ebx = 'ebx'
    ecx = 'ecx'
    edx = 'edx'
    esi = 'esi'
    edi = 'edi'
    ebp = 'ebp'
    esp = 'esp'
    r8d = 'r8d'
    r9d = 'r9d'
    r10d = 'r10d'
    r11d = 'r11d'
    r12d = 'r12d'
    r13d = 'r13d'
    r14d = 'r14d'
    r15d = 'r15d'


class R16(StrEnum):
    ax = 'ax'
    bx = 'bx'
    cx = 'cx'
    dx = 'dx'
    si = 'si'
    di = 'di'
    bp = 'bp'
    sp = 'sp'
    r8w = 'r8w'
    r9w = 'r9w'
    r10w = 'r10w'
    r11w = 'r11w'
    r12w = 'r12w'
    r13w = 'r13w'
    r14w = 'r14w'
    r15w = 'r15w'


class RH(StrEnum):
    ah = 'ah'
    bh = 'bh'
    ch = 'ch'
    dh = 'dh'


class RL(StrEnum):
    al = 'al'
    bl = 'bl'
    cl = 'cl'
    dl = 'dl'
    sil = 'sil'
    dil = 'dil'
    bpl = 'bpl'
    spl = 'spl'
    r8b = 'r8b'
    r9b = 'r9b'
    r10b = 'r10b'
    r11b = 'r11b'
    r12b = 'r12b'
    r13b = 'r13b'
    r14b = 'r14b'
    r15b = 'r15b'


@dataclass
class Register:
    r64: R64
    r32: R32
    r16: R16 | None
    rh: RH | None
    rl: RL | None
    value: int = 0

    def __str__(self):
        return f'{self.r64} = {self.value}'


Registers = [
    Register(r64=R64.rax, r32=R32.eax,r16=R16.ax,rh=RH.ah,rl=RL.al),
    Register(r64=R64.rbx, r32=R32.ebx,r16=R16.bx,rh=RH.bh,rl=RL.bl),
    Register(r64=R64.rcx, r32=R32.ecx,r16=R16.cx,rh=RH.ch,rl=RL.cl),
    Register(r64=R64.rdx, r32=R32.edx,r16=R16.dx,rh=RH.dh,rl=RL.dl),
    Register(r64=R64.rsi, r32=R32.esi,r16=R16.si,rh=None,rl=RL.sil),
    Register(r64=R64.rdi, r32=R32.edi,r16=R16.di,rh=None,rl=RL.dil),
    Register(r64=R64.rbp, r32=R32.ebp,r16=R16.bp,rh=None,rl=RL.bpl),
    Register(r64=R64.rsp, r32=R32.esp,r16=R16.sp,rh=None,rl=RL.spl),
    Register(r64=R64.r8, r32=R32.r8d, r16=R16.r8w,rh=None,rl=RL.r8b),
    Register(r64=R64.r9, r32=R32.r9d, r16=R16.r9w,rh=None,rl=RL.r9b),
    Register(r64=R64.r10, r32=R32.r10d,r16=R16.r10w,rh=None,rl=RL.r10b),
    Register(r64=R64.r11, r32=R32.r11d,r16=R16.r11w,rh=None,rl=RL.r11b),
    Register(r64=R64.r12, r32=R32.r12d,r16=R16.r12w,rh=None,rl=RL.r12b),
    Register(r64=R64.r13, r32=R32.r13d,r16=R16.r13w,rh=None,rl=RL.r13b),
    Register(r64=R64.r14, r32=R32.r14d,r16=R16.r14w,rh=None,rl=RL.r14b),
    Register(r64=R64.r15, r32=R32.r15d,r16=R16.r15w,rh=None,rl=RL.r15b),
]


register_names = set(name for r in Registers for name in (r.r64, r.r32, r.r16, r.rh, r.rl))


class OperandSize(StrEnum):
    byte = 'byte'
    word = 'word'
    dword = 'dword'
    qword = 'qword'


@dataclass
class Operand:
    pass


@dataclass
class NumberOp(Operand):
    value: int


@dataclass
class RegisterOp(Operand):
    value: str


@dataclass
class MemoryOp(Operand):
    base: str | None = None
    index: str | None = None
    scale: int = 0
    displacement: int = 0
