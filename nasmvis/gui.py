from pyray import *

from nasmvis.common import R64
from nasmvis.machine import Machine


BLOCK_HEIGHT = 1050
BLOCK_Y = 100
REG_FONT_SIZE = 20
REGISTER_WIDTH = 300
REGISTER_HEIGHT = 25
REGISTER_X = 200
REGISTER_PAD_Y = 20

MEMORY_X = 600
MEMORY_WIDTH = 200
MEMORY_HEIGHT = 20
MEMORY_PAD_Y = 5
MEMORY_COUNT = 40
MEMORY_FONT_SIZE = 20

INST_COUNT = 30
INST_FONT_SIZE = 20

FONT_SIZE = 30
CODE_X = 1000
CODE_PAD_Y = 5
CODE_HEIGHT = 30
CODE_WIDTH = 800


def draw_register(x: int, y: int, name: str, value: str):
    draw_text(f'{name}: {value}', x, y, REG_FONT_SIZE, BLACK)


def run(code: str, machine: Machine):
    init_window(2000, 1200, "Machine")
    set_target_fps(30)

    step_btn_rect = Rectangle(20, 0, 150, 60)
    continue_btn_rect = Rectangle(20, 100, 150, 60)

    while not window_should_close():
        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
            if check_collision_point_rec(get_mouse_position(), step_btn_rect):
                machine.step()
            if check_collision_point_rec(get_mouse_position(), continue_btn_rect):
                while machine.step():
                    pass

        begin_drawing()
        clear_background(WHITE)

        # draw step button
        draw_rectangle_lines_ex(step_btn_rect, 2, BLACK)

        # draw continue button
        draw_rectangle_lines_ex(continue_btn_rect, 2, BLACK)

        draw_text('Step', int(step_btn_rect.x) + 5, int(step_btn_rect.y) + 5, 30, BLUE)
        draw_text('Continue', int(continue_btn_rect.x) + 5, int(continue_btn_rect.y) + 5, 30, DARKGREEN)

        # draw registers
        draw_text('Registers', REGISTER_X, BLOCK_Y - int(FONT_SIZE * 2), FONT_SIZE, BLACK)
        for i, (name, reg) in enumerate(machine.reg64.items()):
            draw_register(REGISTER_X, BLOCK_Y + (REGISTER_HEIGHT + REGISTER_PAD_Y) * i, name, str(reg.value))

        # draw flags
        for i, (flag, value) in enumerate(machine.flags.items()):
            draw_register(REGISTER_X, BLOCK_Y + (REGISTER_HEIGHT + 20) * (i + len(machine.reg64)), flag, '1' if value else '0')

        draw_rectangle_lines(REGISTER_X - 10, BLOCK_Y - 10, REGISTER_WIDTH + 10 * 2, BLOCK_HEIGHT, LIGHTGRAY)

        # draw stack
        draw_text('Memory', MEMORY_X, BLOCK_Y - int(FONT_SIZE * 2), FONT_SIZE, BLACK)
        for i in range(MEMORY_COUNT):
            memory_addr = len(machine.memory) - i - 1
            draw_text(f'{memory_addr}:  {machine.memory[memory_addr]}', MEMORY_X, BLOCK_Y + i * (MEMORY_HEIGHT + MEMORY_PAD_Y), MEMORY_FONT_SIZE, BLACK)

        # draw memory
        for i in range(MEMORY_COUNT):
            memory_addr = MEMORY_COUNT - 1 - i
            draw_text(f'{memory_addr}:  {machine.memory[memory_addr]}', MEMORY_X + MEMORY_WIDTH, BLOCK_Y + i * (MEMORY_HEIGHT + MEMORY_PAD_Y), MEMORY_FONT_SIZE, BLACK)
        draw_rectangle_lines(MEMORY_X - 10, BLOCK_Y - 10, REGISTER_WIDTH + 10 * 2, BLOCK_HEIGHT, LIGHTGRAY)

        # draw instructions
        draw_text('Code', CODE_X, BLOCK_Y - int(FONT_SIZE * 2), FONT_SIZE, BLACK)
        if machine.rip < len(machine.inst):
            cur_inst_line = machine.inst[machine.rip].line
        else:
            cur_inst_line = machine.inst[-1].line + 1

        lines = code.split('\n')
        start_idx = max(0, cur_inst_line - INST_COUNT // 2)
        for i, l in enumerate(lines[start_idx:start_idx+INST_COUNT]):
            draw_text(f'{i+start_idx}:    {l}', CODE_X, BLOCK_Y + (CODE_HEIGHT + CODE_PAD_Y) * i, INST_FONT_SIZE, BLACK)

        draw_circle(CODE_X - 25, BLOCK_Y + (CODE_HEIGHT + CODE_PAD_Y) * (cur_inst_line - start_idx) + 10, 10, RED)
        draw_rectangle_lines(CODE_X - 10, BLOCK_Y - 10, CODE_WIDTH + 10 * 2, BLOCK_HEIGHT, LIGHTGRAY)

        end_drawing()
    close_window()
