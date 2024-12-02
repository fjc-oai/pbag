"""
- os.environ['PYTHONBREAKPOINT'] to specify the behavior of breakpoint()
- use Pdb().set_trace(frame) to specify the starting frame
- use RemotePdb to allow remote connection and debugging
"""

import logging
from pdb import Pdb
import sys
import os
from remote_pdb import RemotePdb
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "localhost"
PORT = 1234

def fn1(x):
    x += 1
    fn2(x)


def fn2(x):
    x += 2
    fn3(x)


def fn3(x):
    x += 3
    breakpoint()


def set_trace_local():
    logger.info("Entering local set_trace()")
    frame = sys._getframe().f_back
    Pdb().set_trace(frame)


def set_trace_remote():
    logger.info("Entering remote set_trace()")
    frame = sys._getframe().f_back
    RemotePdb(HOST, PORT).set_trace(frame)


def setup_breakpoint(mode):
    if mode == "local":
        os.environ["PYTHONBREAKPOINT"] = "remote_debugger.set_trace_local"
    elif mode == "remote":
        os.environ["PYTHONBREAKPOINT"] = "remote_debugger.set_trace_remote"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="local")
    args = parser.parse_args()
    setup_breakpoint(args.mode)

    x = 10
    fn1(x)


if __name__ == "__main__":
    main()
