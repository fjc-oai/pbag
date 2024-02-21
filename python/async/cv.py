import concurrent.futures
import threading
import time


def test_ordering():
    l = []
    cv = threading.Condition()

    def t1(cv, l):
        with cv:
            cv.wait()
        l.append("t1")

    def t2(cv, l):
        with cv:
            time.sleep(1)
            l.append("t2")
            cv.notify()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = [executor.submit(t, cv, l) for t in (t1, t2)]
        for fut in futs:
            fut.result()
    assert l == ["t2", "t1"]
    print("test_ordering passed")


class MPMCQueue:
    def __init__(self, size):
        self.size = size
        self.queue = []
        self.lock = threading.Lock()
        self.put_cv = threading.Condition(lock=self.lock)
        self.get_cv = threading.Condition(lock=self.lock)

    def put(self, item):
        with self.lock:
            while len(self.queue) >= self.size:
                self.put_cv.wait()
            self.queue.append(item)
            self.get_cv.notify()

    def get(self):
        with self.lock:
            while len(self.queue) == 0:
                self.get_cv.wait()
            item = self.queue.pop(0)
            self.put_cv.notify()
            return item


def test_mpmc_queue():
    q = MPMCQueue(2)
    l = []

    def t1(q, l):
        for i in range(10):
            l.append(q.get())

    def t2(q, l):
        for i in range(10):
            q.put(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = [executor.submit(t, q, l) for t in (t1, t2)]
        for fut in futs:
            fut.result()
    assert l == list(range(10))
    print("test_mpmc_queue passed")


def main():
    test_ordering()
    test_mpmc_queue()


if __name__ == "__main__":
    main()
