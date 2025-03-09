import random

import cpython_thread_profiler as profiler


def fn_a():
    return random.randint(0, 100)

def fn_b():
    return fn_a() + 1

def fn_c():
    return fn_b() + 1

profiler.set_profiler()

fn_c()

res = profiler.dump_callstacks()
print(res)
profiler.unset_profiler()