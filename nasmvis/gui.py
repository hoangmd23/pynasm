from pyray import *

from nasmvis.common import R64
from nasmvis.machine import Machine


REGISTER_WIDTH = 300
REGISTER_HEIGHT = 35
FONT_SIZE = 30
CODE_X = 1000
CODE_Y = 20
CODE_HEIGHT = 30


def draw_register(x: int, y: int, name: str, value: str):
    draw_rectangle_lines(x, y, REGISTER_WIDTH, REGISTER_HEIGHT, GRAY)
    draw_text(f'{name}: {value}', x, y, FONT_SIZE, BLUE)


def run(code: str, machine: Machine):
    init_window(2000, 1200, "Machine")
    set_target_fps(30)
    step_btn_rect = Rectangle(0, 0, 100, 100)
    while not window_should_close():
        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
            if check_collision_point_rec(get_mouse_position(), step_btn_rect):
                machine.step()

        begin_drawing()
        clear_background(WHITE)
        # draw step button
        draw_rectangle_rec(step_btn_rect, BLUE)

        # draw registers
        for i, (name, reg) in enumerate(machine.reg64.items()):
            draw_register(200, 20 + (REGISTER_HEIGHT + 20) * i, name, str(reg.value))

        # draw flags
        for i, (flag, value) in enumerate(machine.flags.items()):
            draw_register(200, 20 + (REGISTER_HEIGHT + 20) * (i + len(machine.reg64)), flag, str(value))

        # draw stack
        for i in range(50):
            memory_addr = len(machine.memory) - i - 1
            draw_text(f'{memory_addr}:  {machine.memory[memory_addr]}', 600, i * 20, 20, BLACK)
        draw_rectangle(600 - 30, 20 * (len(machine.memory) - machine.reg64[R64.rsp].value - 1), 20, 20, RED)

        # draw memory
        for i in range(50):
            memory_addr = 50 - 1 - i
            draw_text(f'{memory_addr}:  {machine.memory[memory_addr]}', 800, i * 20, 20, BLACK)

        # draw instructions
        for i, l in enumerate(code.split('\n')):
            draw_text(f'{i}:    {l}', CODE_X, CODE_Y + CODE_HEIGHT * i, 20, BLACK)

        cur_inst = machine.inst[machine.rip]
        draw_rectangle(CODE_X - 30, CODE_Y + CODE_HEIGHT * cur_inst.line, 20, 20, RED)

        end_drawing()
    close_window()
