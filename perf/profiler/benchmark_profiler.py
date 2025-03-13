import sys
import time

import cprofiler

stats = {}


def record_call(func_name):
    global stats
    stats[func_name] = stats.get(func_name, 0) + 1


def profiler(frame, event, arg):
    func_name = frame.f_code.co_name
    line_no = frame.f_lineno

    if event == "call":
        record_call(f"{func_name}:{line_no}")
    elif event == "return":
        pass
    elif event == "exception":
        pass
    return profiler


def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def task():
    for i in range(1000):
        fibonacci(20)


def duration(func, n_trials=3):
    st = time.perf_counter()
    for _ in range(n_trials):
        func()
    et = time.perf_counter()
    return (et - st) / n_trials


def main():
    dur = duration(task)
    print(f"Baseline: {dur:.2f} seconds")

    sys.setprofile(profiler)
    dur = duration(task)
    print(f"Python profiler: {dur:.2f} seconds")
    sys.setprofile(None)
    print(stats)

    cprofiler.set_profiler()
    dur = duration(task)
    print(f"C++ profiler: {dur:.2f} seconds")
    cprofiler.unset_profiler()
    c_stats = cprofiler.dump_stats()
    print(c_stats)


if __name__ == "__main__":
    main()
