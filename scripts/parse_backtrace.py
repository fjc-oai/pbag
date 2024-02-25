#!/usr/bin/env python3

# chmod +x parse_bt.py
# cp parse_bt.py /usr/local/bin/parse_bt
# echo "...your bt..." | parse_bt
import sys
import argparse

def main():
    argparser = argparse.ArgumentParser(description="Parse a backtrace")
    argparser.add_argument("trim", type=int, default=0) 
    args = argparser.parse_args()

    traceback_input = sys.stdin.read()
    traceback_lines = traceback_input.split("\n")
    for line in traceback_lines:
        if not line.startswith("->"):
            if args.trim > 0:
                parts = line.split("/")
                parts = parts[args.trim:]
                line = "/".join(parts)
            print(line)


if __name__ == "__main__":
    main()
