import threading
import time


def cpu_bound_task():
    x = 0 
    for _ in range(10000000):
        for _ in range(10):
            x+=1

def wrapper(idx):
    thread_id = threading.get_native_id()
    cpu_time_start = time.thread_time()
    wall_time_start = time.perf_counter()
    cpu_bound_task()
    cpu_time = time.thread_time() - cpu_time_start
    wall_time = time.perf_counter() - wall_time_start
    time.sleep(idx * 0.1)
    print(f"thread {thread_id}")
    print(f"cpu time: {cpu_time}, start: {cpu_time_start}, end: {cpu_time}")
    print(f"wall time: {wall_time}, start: {wall_time_start}, end: {wall_time}")



threads = []
for idx in range(3):
    t = threading.Thread(target=wrapper, args=(idx,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()