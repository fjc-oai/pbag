#!/usr/bin/env python3

# chmod +x parse_backtrace.py
# cp parse_backtrace.py /usr/local/bin/parse_bt
# echo "...your bt..." | parse_bt
import argparse
import sys


def _process_frame(line, args):
    if line.startswith("->"):
        return ""
    if args.drop_frame:
        key_works = [
            "lib/python3",
        ]
        for key in key_works:
            if key in line:
                return ""
    if args.trim_prefix:
        parts = line.split("/")
        line = "/".join(parts[4:])
    return line


def main():
    argparser = argparse.ArgumentParser(description="Parse a backtrace")
    argparser.add_argument("--drop-frame", action="store_true", default=True)
    argparser.add_argument("--trim-prefix", action="store_true", default=True)
    args = argparser.parse_args()

    traceback_input = sys.stdin.read()
    print("\n")
    traceback_lines = traceback_input.split("\n")
    for line in traceback_lines:
        line = _process_frame(line, args)
        if line:
            print(line)
    print("\n")


if __name__ == "__main__":
    main()
