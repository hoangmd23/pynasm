import sys

from nasmsim.machine import Machine
from nasmsim.parser import parse_instructions


def print_help():
    print("Usage: python app.py <path_to_asm>")
    print("Example: python app.py main.asm")
    print("\nOptions:")
    print("  -h, --help     Show help message")


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    asm_path = sys.argv[1]

    from nasmsim.gui import run

    with open(asm_path, 'r') as f:
        code = f.read()
    start_addr, inst, data, data_labels, bss_size = parse_instructions(code, True)

    machine = Machine()
    machine.load_inst_and_data(start_addr, inst, data, data_labels, bss_size)

    run(code, machine)

if __name__ == '__main__':
    main()
