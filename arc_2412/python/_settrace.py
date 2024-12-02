import sys
import time
from collections import defaultdict
from tabulate import tabulate


class Profiler:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._events = []

    def add_event(self, event, fn_name, tp):
        self._events.append((event, fn_name, tp))

    def profile(self):
        event_stack = []
        event_dur = defaultdict(list)
        for event, fn_name, tp in self._events:
            if event == "call":
                event_stack.append((fn_name, tp))
            elif event == "return":
                pre_event = event_stack.pop()
                assert pre_event[0] == fn_name
                dur = tp - pre_event[1]
                event_dur[fn_name].append(dur)
        data = [
            ["fn_name", "n_calls", "avg_dur"],
        ]
        for fn_name, durs in event_dur.items():
            data.append([fn_name, len(durs), sum(durs) / len(durs)])
        print(tabulate(data, headers="firstrow"))


def trace(frame, event, arg):
    if event in ["call", "return"]:
        tp = time.perf_counter()
        Profiler.get().add_event(event, frame.f_code.co_name, tp)
    return trace


def f1(x):
    time.sleep(1)
    return f2(x) + f2(x)


def f2(x):
    time.sleep(1)
    return f3(x) + f3(x)


def f3(x):
    time.sleep(1)
    return x + 3


def main():
    sys.settrace(trace)
    f1(10)
    sys.settrace(None)
    Profiler.get().profile()


if __name__ == "__main__":
    main()
