import concurrent.futures
import random
import threading
import time
from dataclasses import dataclass


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


@dataclass
class State:
    lock = threading.Lock()
    is_racing = False
    n_players = 0
    group_conditions = {}
    official_condition = None


def player_thread(player_id, group_id, state):
    with state.lock:
        if group_id not in state.group_conditions:
            state.group_conditions[group_id] = threading.Condition(state.lock)
        group_condition = state.group_conditions[group_id]
        while not state.is_racing:
            group_condition.wait()
        state.n_players += 1
    print(f"Group {group_id}: player {player_id} starts")
    time.sleep(random.uniform(2, 5))
    print(f"Group {group_id}: player {player_id} finishes")
    with state.lock:
        state.n_players -= 1
        if state.n_players == 0:
            state.is_racing = False
            state.official_condition.notify()


def official_thread(state, n_groups):
    for group_id in range(n_groups):
        with state.lock:
            state.official_condition = threading.Condition(state.lock)
            while state.is_racing:
                state.official_condition.wait()
            s = f"Official: group {group_id} starts racing"
            print("*" * len(s))
            print(s)
            print("*" * len(s))
            state.is_racing = True
            group_condition = state.group_conditions[group_id]
            group_condition.notify_all()

def test_race():
    state = State()
    N_PLAYERS = 12
    N_GROUPS = 3
    PLAYERS_PER_GROUP = N_PLAYERS // N_GROUPS
    futs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_PLAYERS + 1) as executor:
        for i in range(N_PLAYERS):
            futs.append(executor.submit(player_thread, i, i // PLAYERS_PER_GROUP, state))
        futs.append(executor.submit(official_thread, state, N_GROUPS))
    for fut in futs:
        fut.result()
    print("test_race passed")

def worker_thread(idx, lock, state):
    with lock:
        cv = threading.Condition(lock)
        state[idx] = cv
        cv.wait()
    print(f"worker {idx} starts")

def test_shared_lock():
    lock = threading.Lock()
    state = {}
    futs = []
    N_WORKERS = 5
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS)
    for i in range(N_WORKERS):
        futs.append(executor.submit(worker_thread, i, lock, state))
    print("All workers are created and waiting")
    wakeup_list = list(range(N_WORKERS))
    import random
    random.shuffle(wakeup_list)
    print(f"Waking up workers in order: {wakeup_list}")
    for idx in wakeup_list:
        with lock:
            state[idx].notify()
            time.sleep(1)
    for fut in futs:
        fut.result()
    print("test_shared_lock passed")


def main():
    # test_ordering()
    # test_mpmc_queue()
    # test_race()
    test_shared_lock()


if __name__ == "__main__":
    main()
