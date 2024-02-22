#!/usr/bin/env python3

# chmod +x parse_bt.py
# cp parse_bt.py /usr/local/bin/parse_bt
# echo "...your bt..." | parse_bt
import sys


def main():
    traceback_input = sys.stdin.read()
    traceback_lines = traceback_input.split("\n")
    for line in traceback_lines:
        if not line.startswith("->"):
            print(line)


if __name__ == "__main__":
    main()
