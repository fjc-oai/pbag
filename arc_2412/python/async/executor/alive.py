"""
- Python thread doesn't automatically propagate exception to the 
    main thread. Very different from C++!
- A thread can die silently without any exception info! Always
    manually check alive from the main thread if you care. 
- Customize the thread class if thread exception info is needed.
"""

import random
import threading
import time


class ThreadWithException(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e


def worker(idx, event):
    print(f"Worker {idx} start running...")
    while not event.is_set():
        if random.random() < 0.1:
            print(f"Worker {idx} is raising!")
            raise Exception(f"Worker {idx} raises!")
        time.sleep(1)
    print(f"Worker {idx} completes!")


def main():
    event = threading.Event()
    N_WORKERS = 3
    N_ITER = 10
    threads = [ThreadWithException(target=worker, args=(i, event)) for i in range(N_WORKERS)]
    for t in threads:
        t.start()

    alive_threads = [t for t in threads if t.is_alive()]
    for iter in range(N_ITER):
        print(f"Check alive at iter {iter}...")
        for thread in alive_threads:
            if not thread.is_alive():
                print(f"Thread {thread.ident} is dead at iter {iter} with exception {thread.exc}!")
        alive_threads = [t for t in threads if t.is_alive()]
        time.sleep(1)

    print(f"Stopping all workers after {N_ITER} iterations...")
    event.set()
    for t in threads:
        t.join()
    print("All workers stopped!")


if __name__ == "__main__":
    main()
