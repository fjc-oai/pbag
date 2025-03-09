import time

import burn_module


def burn_cpu(dur):
    cur = time.time()
    x = 0
    while time.time() - cur < dur:
        x += 1
        x %= 1000


def cpu_task(idx):
    thread_id = threading.get_native_id()
    cpu_time_start = time.thread_time()
    wall_time_start = time.perf_counter()
    # burn_cpu(1)
    burn_module.burn(1)
    cpu_time = time.thread_time() - cpu_time_start
    wall_time = time.perf_counter() - wall_time_start
    time.sleep(idx * 0.1)
    print(f"thread {thread_id}")
    print(f"cpu time: {cpu_time}, start: {cpu_time_start}, end: {cpu_time}")
    print(f"wall time: {wall_time}, start: {wall_time_start}, end: {wall_time}")

import threading

threads = []
for idx in range(3):
    t = threading.Thread(target=cpu_task, args=(idx,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
