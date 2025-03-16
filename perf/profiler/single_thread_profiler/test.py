import cprofiler
import time
from tqdm import tqdm
cprofiler.set_profiler()


def task(duration):
    time.sleep(duration * 0.1)


def f1():
    f1_1()
    f1_2()


def f1_1():
    task(2)


def f1_2():
    task(3)


def f2():
    f2_1()
    f2_2()
    f2_3()


def f2_1():
    task(3)


def f2_2():
    task(4)


def f2_3():
    task(5)


def f3():
    f3_1()
    f3_2()
    f3_3()


def f3_1():
    task(4)


def f3_2():
    task(5)


def f3_3():
    task(6)


def foo():
    f1()
    f2()
    f3()


foo()

cprofiler.unset_profiler()
res = cprofiler.dump_stats()

def dump_line(stacktrace, duration, pos_2_file_name, pos_2_func_name, full_file_name=False):
    lines = []
    for frame in stacktrace:
        file_name = pos_2_file_name[frame[0]]
        if not full_file_name:
            file_name = file_name.split("/")[-1]
        func_name = pos_2_func_name[frame[1]]
        line_no = frame[2]
        line = f"{file_name}:{func_name}({line_no})"
        lines.append(line) 
    return ";".join(lines) + f" {duration}"

def dump_stats_to_file(res, file_path):
    pos_2_file_name = res["pos_2_file_name"]
    pos_2_func_name = res["pos_2_func_name"]
    stacktraces = res["stacktraces"]
    durations = res["durations"]
    assert len(stacktraces) == len(durations), f"stacktraces and durations have different lengths: {len(stacktraces)} != {len(durations)}"
    with open(file_path, "w") as f:
        for stacktrace, duration in tqdm(zip(stacktraces, durations), total=len(stacktraces)):
            line = dump_line(stacktrace, duration, pos_2_file_name, pos_2_func_name)
            f.write(line)
            f.write("\n")

dump_stats_to_file(res, "stats.txt")
