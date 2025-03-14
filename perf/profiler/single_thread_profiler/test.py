import cprofiler
import time

cprofiler.set_profiler()


def task(duration):
    time.sleep(duration * 0.1)


def f1():
    f1_1()
    f1_2()


def f1_1():
    task(1)


def f1_2():
    task(2)


# def f2():
#     f2_1()
#     f2_2()
#     f2_3()


# def f2_1():
#     task(1)


# def f2_2():
#     task(2)


# def f2_3():
#     task(3)


# def f3():
#     f3_1()
#     f3_2()
#     f3_3()


# def f3_1():
#     task(1)


# def f3_2():
#     task(2)


# def f3_3():
#     task(3)




def foo():
    f1()
    # f2()
    # f3()


foo()

cprofiler.unset_profiler()


def dump_stats_to_file(stats, file_path):
    with open(file_path, "w") as f:
        for stacktrace, count in stats.items():
            f.write(f"{stacktrace}: {count}\n")
    print(f"Stats dumped to {file_path}")


res = cprofiler.dump_stats()
dump_stats_to_file(res, "stats.txt")
