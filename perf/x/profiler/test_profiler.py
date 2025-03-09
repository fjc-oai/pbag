import argparse
import random
import threading
import time

import cpython_thread_profiler as profiler


def inner_function(n):
    """A recursive function that creates a deeper call stack."""
    time.sleep(random.uniform(0.01, 0.05))
    if n > 0:
        inner_function(n - 1)
    return

def extra_function1():
    """Additional function to generate a call stack."""
    print("Entering extra_function1")
    inner_function(2)
    print("Exiting extra_function1")

def extra_function2(n):
    """Calls extra_function1 multiple times."""
    for i in range(n):
        print(f"extra_function2: iteration {i}")
        extra_function1()

def extra_function3(n):
    """Recursive function to build a more complex call stack."""
    if n <= 0:
        return
    print(f"In extra_function3: recursion level {n}")
    extra_function1()
    extra_function3(n - 1)
    print(f"Exiting extra_function3: recursion level {n}")

def complex_routine(n):
    """Routine that calls different functions in sequence for a complex call stack."""
    print("Starting complex_routine")
    inner_function(n)
    extra_function2(2)
    extra_function3(2)
    print("Finished complex_routine")

def thread_worker(name, iterations, depth):
    """Worker function that calls multiple functions to generate various call stacks."""
    print(f"Thread {name} (ID: {threading.get_ident()}) starting")
    for i in range(iterations):
        print(f"Thread {name}, iteration {i} starting")
        inner_function(depth)
        extra_function1()
        extra_function2(1)
        extra_function3(1)
        complex_routine(depth)
        print(f"Thread {name}, iteration {i} finished")
        time.sleep(random.uniform(0.05, 0.15))
    print(f"Thread {name} (ID: {threading.get_ident()}) finished")

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded Profiler Test with Additional Functions")
    parser.add_argument("--threads", type=int, default=5,
                        help="Number of worker threads")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per thread")
    parser.add_argument("--depth", type=int, default=3,
                        help="Recursion depth for inner_function")
    args = parser.parse_args()

    print("Enabling profiler...")
    profiler.set_profiler()

    threads = []
    for i in range(args.threads):
        name = f"Worker-{i}"
        t = threading.Thread(target=thread_worker, args=(name, args.iterations, args.depth))
        threads.append(t)
        t.start()

    # Wait for all threads to finish.
    for t in threads:
        t.join()

    print("Disabling profiler...")
    profiler.unset_profiler()

    # Retrieve and print aggregated call stacks along with their invocation counts.
    dump = profiler.dump_callstacks()
    print("\nDumped call stacks and their invocation counts:")
    for callstack, count in dump.items():
        print(f"{callstack} : {count}")

if __name__ == '__main__':
    main() 