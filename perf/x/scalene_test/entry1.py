import concurrent.futures
import time

from compute import count_primes
from io_bound import read_large_file, write_large_file
from memory import generate_large_array


def main():
    num_threads = 16  # Adjust as needed

    # Threaded CPU-bound task
    prime_target = 1000000

    # Threaded Memory-bound task
    array_size = 2000  # Creates a 2000x2000 array

    # Threaded I/O-bound task
    filename = "test_file.txt"
    num_lines = 50000

    for i in range(1000):
        for j in range(1000):
            x = i * j

    print(x)
    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        # CPU-bound tasks
        for i in range(num_threads //2 ):
            futures.append(executor.submit(count_primes, prime_target, i))

        # Memory-bound tasks
        for i in range(num_threads //2 ):
            future = executor.submit(generate_large_array, array_size, i)
            futures.append(future)

        # I/O-bound tasks (write then read)
        for i in range(num_threads //2 ):
            futures.append(executor.submit(write_large_file, f"{filename}_{i}.txt", num_lines, i))

        for i in range(num_threads //2 ):
            futures.append(executor.submit(read_large_file, f"{filename}_{i}.txt", i))

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks are completed

if __name__ == "__main__":
    main()