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


def _process_frame_logger_format(line, args):
    # breakpoint()
    if not line:
        return ""
    pos = line.find(")")
    assert pos != -1, f"Invalid line format: {line}"
    line = line[pos + 1 :]
    line = line.strip()
    if not line.startswith("File"):
        return ""

    if args.drop_frame:
        key_works = [
            "lib/python3",
        ]
        for key in key_works:
            if key in line:
                return ""

    line = line[len('File "') :]
    pos = line.find('"')
    path = line[:pos]
    if args.trim_prefix:
        parts = path.split("/")
        path = "/".join(parts[4:])
    return path + line[pos + 1 :]


def main():
    argparser = argparse.ArgumentParser(description="Parse a backtrace")
    argparser.add_argument("--drop-frame", action="store_true", default=True)
    argparser.add_argument("--trim-prefix", action="store_true", default=True)
    argparser.add_argument("--logger-output", action="store_true", default=False)
    args = argparser.parse_args()

    traceback_input = sys.stdin.read()
    print("\n")
    traceback_lines = traceback_input.split("\n")
    for line in traceback_lines:
        if args.logger_output:
            line = _process_frame_logger_format(line, args)
        else:
            line = _process_frame(line, args)
        if line:
            print(line)
    print("\n")


if __name__ == "__main__":
    main()
