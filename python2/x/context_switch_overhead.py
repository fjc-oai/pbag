"""
Running benchmark 1/6
1 threads: 0.407460 seconds
Running benchmark 2/6
10 threads: 0.392594 seconds
Running benchmark 3/6
50 threads: 0.419609 seconds
Running benchmark 4/6
100 threads: 0.456290 seconds
Running benchmark 5/6
1000 threads: 1.097902 seconds
Running benchmark 6/6
10000 threads: 1.399982 seconds
"""

import math
import threading
import time


def worker(n):
    result = 0
    for i in range(n):
        result += math.sqrt(i * (math.pi)) + math.cos(i) + math.sin(i) + math.tan(i)
        result *= math.pi
        tmp = str(result)
        result = float(tmp)
    return result


def benchmark(num_threads, n_work):
    threads = []
    work_per_thread = n_work // num_threads

    for _ in range(num_threads):
        thread = threading.Thread(target=worker, args=(work_per_thread,))
        threads.append(thread)

    start_time = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    return end_time - start_time


def test_context_switch_overhead():
    n_thread_list = [1, 10, 50, 100, 1000, 10000]
    n_work = 10**6

    for _ in range(3):
        for idx, n_thread in enumerate(n_thread_list):
            print(f"Running benchmark {idx+1}/{len(n_thread_list)}")
            exec_time = benchmark(n_thread, n_work)
            print(f"{n_thread} threads: {exec_time:.6f} seconds")


if __name__ == "__main__":
    test_context_switch_overhead()
