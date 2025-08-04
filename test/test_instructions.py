import os
from pathlib import Path

import yaml

from pynasm.common import OperandSize
from pynasm.machine import Machine
from pynasm.parser import parse_instructions


def load_test_cases(path: str):
    with open(Path(__file__).parent / 'data' / path) as f:
        return yaml.safe_load(f)


def prepare_machine(code: str) -> Machine:
    machine = Machine()
    machine.load_inst_and_data(*parse_instructions(code, False))
    return machine


def execute_tests(test_data_path: str, update_flags=False):
    print(f'Executing tests on {test_data_path}')
    test_cases = load_test_cases(test_data_path)
    for case_id, tc in enumerate(test_cases):
        asm = tc['asm']
        expected = tc['expected']

        machine = prepare_machine(asm)
        if update_flags:
            machine.set_flags(update_flags)
        while machine.step():
            pass

        for exp_type, exp in expected.items():
            match exp_type:
                case 'register':
                    for reg, exp_value in exp.items():
                        if isinstance(exp_value, str) and exp_value.startswith('0x'):
                            exp_value = int(exp_value, 16)
                        actual_value = machine.get_register(reg)
                        assert actual_value == exp_value, f'Test case {case_id} has failed\n{asm}\nRegister {reg}\nActual value: {actual_value}\nExpected value: {exp_value}'
                case 'memory':
                    for addr, exp_value in exp.items():
                        actual_value = machine.read_memory(addr, OperandSize.byte)
                        assert actual_value == exp_value, f'Test case {case_id} has failed\n{asm}\nMemory at address {addr}\nActual value: {actual_value}\nExpected value: {exp_value}'
                case 'flags':
                    for flag, exp_value in exp.items():
                        actual_value = machine.read_flag(flag)
                        assert actual_value == exp_value, f'Test case {case_id} has failed\n{asm}\nFlag {flag}\nActual value: {actual_value}\nExpected value: {exp_value}'
                case _:
                    raise NotImplementedError(f'{exp_type} is not supported in expected values')


def run_all_tests():
    for f in os.listdir('test/data'):
        execute_tests(f, False)
        if f == 'test_jmp.yaml':
            execute_tests(f, True)
