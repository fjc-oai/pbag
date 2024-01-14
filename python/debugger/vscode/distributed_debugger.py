"""
- enable remote debugging on ray actors
- TOOD: driver can be a integrated debugger to interactively 
    connect with different alive debuggers on different actors
- ISSUE: it seems not all actors are listening on the port though..
"""
import ray
import logging
import debugpy
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PORT = 1234


def get_port():
    return int(os.environ.get("MY_PORT", PORT))


def set_trace():
    debugpy.listen(("localhost", get_port()))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    debugpy.breakpoint()


@ray.remote
class Actor:
    def __init__(self, rank):
        self.rank = rank
        os.environ["MY_PORT"] = str(PORT + self.rank)
        os.environ["PYTHONBREAKPOINT"] = "distributed_debugger.set_trace"

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
    N = 2
    actors = [Actor.remote(i) for i in range(N)]
    futs = [actor.fn1.remote(10) for actor in actors]
    ray.get(futs)


if __name__ == "__main__":
    main()
