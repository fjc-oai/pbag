# https://realpython.com/python-flush-print-output/

import time


def line_buffered_in_interact_env(add_newline=True, force_flush=False):
    # python print_buffer.py
    for i in range(3, 0, -1):
        print(i, end=' ' if not add_newline else None, flush=force_flush)
        time.sleep(1)
    print('Go!')

def main():
    # 1) In interactive environment, the output is line buffered by default
    # >>> python print_buffer.py
    line_buffered_in_interact_env()

    # 2) In interactive environment, when disable newline, the output is block-buffered and printed all at once 
    # >>> python print_buffer.py
    line_buffered_in_interact_env(add_newline=False)

    # 3) In non-interactive environment, the output is block-buffered by default
    # python print_buffer.py | cat
    line_buffered_in_interact_env()

    # 4) force flushing the buffer always prints the output immediately
    # python print_buffer.py | cat
    line_buffered_in_interact_env(force_flush=True)

if __name__ == '__main__':
    main()

