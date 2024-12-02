from threading import Event, Thread, Lock


def worker(idx, wait_evt, set_evt):
    print(f"Worker {idx} started")
    for i in range(5):
        wait_evt.wait()
        print(f"Worker {idx}: {i}")
        wait_evt.clear()
        set_evt.set()
    print(f"Worker {idx} done")


class ToyEvent:
    def __init__(self):
        self._lock = Lock()
        self._lock.acquire()

    def wait(self):
        self._lock.acquire()
        self._lock.release()

    def set(self):
        self._lock.release()

    def clear(self):
        self._lock.acquire()


def get_event(use_toy):
    if use_toy:
        return ToyEvent()
    else:
        return Event()


def test_event(use_toy):
    evt1 = get_event(use_toy)
    evt2 = get_event(use_toy)

    t1 = Thread(target=worker, args=(1, evt1, evt2))
    t2 = Thread(target=worker, args=(2, evt2, evt1))

    t1.start()
    t2.start()

    evt1.set()

    t1.join()
    t2.join()


if __name__ == "__main__":
    for use_toy in [True, False]:
        print(f"Using toy event: {use_toy}")
        test_event(use_toy)
