"""
A demonstration of using C++ code in Python with pybind11.

Also shows Python intepreter won't have non-voluntary context switch when a
thread is running native code. It does have non-voluntary context switch when a
thread is running Python code though, by injecting check-interval when executing
bytecode.

Example output:
> py tests/test.py
    Now printing every second: 0
    Sleeping 5s to let the thread print
    Now printing every second: 1
    Now printing every second: 2
    Now printing every second: 3
    Now printing every second: 4
    Now calling native code
    Now printing every second: 5
    Time taken for adding 3000 elements: 20.000398874282837
    Now printing every second: 6
    Now printing every second: 7
    Now printing every second: 8
    Now printing every second: 9
    Done
"""
import threading
import time

import nccl_comm.ops


def print_every_sec():
    i = 0 
    while True:
        print(f"Now printing every second: {i}")    
        i += 1
        time.sleep(1)
        if i == 10:
            break

t = threading.Thread(target=print_every_sec)
t.start()
print(f"Sleeping 5s to let the thread print")
time.sleep(5)

print(f"Now calling native code")
cur = time.time()
nccl_comm.ops.busy_loop(20000)
print(f"Time taken for adding 3000 elements: {time.time() - cur}")

t.join()
print(f"Done")

