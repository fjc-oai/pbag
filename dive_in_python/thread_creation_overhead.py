"""
On-the-fly thread creation time: 16.1233 seconds
Thread pool executor time: 14.7124 seconds
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Number of tasks to execute
NUM_TASKS = 1024*32

# Function representing a CPU-bound task
def task():
    # Simulate CPU-bound work by calculating prime numbers
    count = 0
    for num in range(2, 1000):
        prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                prime = False
                break
        if prime:
            count += 1

# Benchmark: Creating threads on the fly
def benchmark_on_the_fly():
    threads = []
    start_time = time.time()
    for _ in range(NUM_TASKS):
        t = threading.Thread(target=task)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return time.time() - start_time

# Benchmark: Using a global ThreadPoolExecutor
def benchmark_thread_pool():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(lambda _: task(), range(NUM_TASKS))
    return time.time() - start_time

if __name__ == '__main__':
    # Benchmark on-the-fly thread creation
    time_on_the_fly = benchmark_on_the_fly()
    print(f"On-the-fly thread creation time: {time_on_the_fly:.4f} seconds")

    # Benchmark global ThreadPoolExecutor
    time_thread_pool = benchmark_thread_pool()
    print(f"Thread pool executor time: {time_thread_pool:.4f} seconds")