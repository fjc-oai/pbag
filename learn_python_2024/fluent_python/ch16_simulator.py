import dataclasses
import random
from typing import Literal

import tabulate


def random_duration() -> int:
    return random.randint(1, 10)

@dataclasses.dataclass
class DriverState:
    id: int 
    n_trips: int
    n_completed_trips: int = 0
    ts_last_event: int = 0
    duration_cur_event: int = dataclasses.field(init=False)
    state: Literal["home", "waiting", "driving", "back"] = "home"

    def __post_init__(self) -> None:
        self.duration_cur_event = random_duration()

    def tick(self, ts: int) -> tuple[int, int, str] | None:
        if ts - self.ts_last_event == self.duration_cur_event and self.state != "back":
            return self._transition(ts)
        return None

    def _transition(self, ts: int) -> tuple[int, int, str]:
        self.ts_last_event = ts
        self.duration_cur_event = random_duration()
        if self.state == "home":
            self.state = "waiting"
            return (self.id, ts, "leave garage")
        elif self.state == "waiting":
            self.state = "driving"
            return (self.id, ts, "pick up passenger")
        elif self.state == "driving":
            self.n_completed_trips += 1
            if self.n_completed_trips == self.n_trips:
                self.state = "back"
                return (self.id, ts, "drop off && going home")
            else:
                self.state = "waiting"
                return (self.id, ts, "drop off passenger")
        assert False, f"not reachable {self.state}"

def scheduler(n_drivers: int, n_trips: int):
    events = []
    drivers = [DriverState(i, n_trips) for i in range(n_drivers)]
    ts = 0
    while any(d.state != "back" for d in drivers):
        for d in drivers:
            event = d.tick(ts)
            if event:
                events.append(event)
        ts += 1
    return events

def main():
    events = scheduler(3, 5)
    events = [(f"driver_{d}", ts, action) for d, ts, action in events]
    table = tabulate.tabulate(events, headers=["driver", "timestamp", "action"])
    print(table)

if __name__ == "__main__":
    main()

            
