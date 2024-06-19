# Not really a good example but kind of demonstrates how does coroutines work
import heapq
import random

import tabulate


def random_duration() -> int:
    return random.randint(1, 10)


def driver(id, n_trips):
    ts = yield (id, 0, "leave garage")
    for i in range(n_trips):
        ts = yield (id, ts, "pick up passenger")
        ts = yield (id, ts, "drop off passenger")
    yield (id, ts, "drop off && going home")


def scheduler(n_drivers: int, n_trips: int):
    pq = []
    drivers = [driver(i, n_trips) for i in range(n_drivers)]
    for d in drivers:
        id, ts, action = next(d)
        heapq.heappush(pq, (ts, (id, ts, action)))
    data = []
    while pq:
        _, (id, ts, action) = heapq.heappop(pq)
        data.append((f"driver_{id}", ts, action))
        try:
            dur = random_duration()
            id, ts, action = drivers[id].send(ts + dur)
            heapq.heappush(pq, (ts, (id, ts, action)))
        except StopIteration:
            pass

    table = tabulate.tabulate(data, headers=["driver", "timestamp", "action"])
    print(table)


def main():
    scheduler(3, 5)


if __name__ == "__main__":
    main()
