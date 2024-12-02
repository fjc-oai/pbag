"""
- enable remote debugging on ray actors
- TOOD: driver can be a integrated debugger to interactively 
    connect with different alive debuggers on different actors
- ISSUE: it seems not all actors are listening on the port though..
"""
import ray
import logging
import sys
import os
from remote_pdb import RemotePdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PORT = 1234


def get_port():
    return int(os.environ.get("MY_PORT", PORT))


def set_trace_remote():
    frame = sys._getframe().f_back
    RemotePdb("localhost", get_port()).set_trace(frame)


@ray.remote
class Actor:
    def __init__(self, rank):
        self.rank = rank
        os.environ["MY_PORT"] = str(PORT + self.rank)
        os.environ["PYTHONBREAKPOINT"] = "distributed_debugee.set_trace_remote"

    def get_port(self):
        return get_port()

    def fn1(self, x):
        x += 1
        return self.fn2(x)

    def fn2(self, x):
        x += 2
        breakpoint()
        return x


def main():
    ray.init()
    N = 3
    actors = [Actor.remote(i) for i in range(N)]
    futs = [actor.fn1.remote(10) for actor in actors]
    import time

    time.sleep(1)

    futs = [actor.get_port.remote() for actor in actors]
    res = ray.get(futs)
    print(f"Actors are listening on ports: {res}")
    print("Use telnet localhost <port> to connect to the actor.")
    print("Use socket to send commands to the actor.")
    input("Waiting for you to connect to the actor...")


if __name__ == "__main__":
    main()
