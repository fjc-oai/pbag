import collections
import heapq
import time
from typing import Any, Callable, Coroutine, Literal, Sequence


class EventLoop:
    def __init__(self):
        self._ready: collections.deque[Handle] = collections.deque()
        self._waiting: list[Handle] = []
        self._scheduled: list[TimerHandle] = []
        self._stopping = False

    def create_task(self, coro: Coroutine) -> "Task":
        task = Task(coro, self)
        return task

    def call_soon(self, callback: Callable, *args) -> "Handle":
        handle = Handle(callback, args, self)
        self._ready.append(handle)
        return handle

    def run_until_complete(self, task: "Task") -> Any:
        self.run_forever()
        return task.result()

    def run_forever(self) -> None:
        # TODO: proper error handling
        while self._ready or self._waiting:
            self._run_once()
            # TODO: support stopping
            if self._stopping:
                break

    def _run_once(self) -> None:
        while self._scheduled and self._scheduled[0].when() <= self.time():
            timer_handle = heapq.heappop(self._scheduled)
            timer_handle._scheduled = False # why?
            self._ready.append(timer_handle)

        while self._ready:
            handle = self._ready.popleft()
            # TODO: support handle is cancelled
            handle._run()

    def create_future(self) -> "Future":
        return Future(self)
    
    def call_later(self, delay: float, callback: Callable, *args) -> "TimerHandle":
        assert delay >= 0
        return self.call_at(self.time() + delay, callback, *args)

    def call_at(self, when: float, callback: Callable, *args) -> "TimerHandle":
        timer = TimerHandle(when, callback, args, self)
        heapq.heappush(self._scheduled, timer)
        timer._scheduled = True
        return timer

    def time(self):
        return time.monotonic()


_event_loop = None


def get_event_loop() -> EventLoop:
    global _event_loop
    if _event_loop is None:
        _event_loop = EventLoop()
    return _event_loop


class Future:
    # Class variables serving as defaults for instance variables.
    _state: Literal["PENDING", "CANCELLED", "FINISHED"]
    _result = None
    # TODO: support callback

    def __init__(self, loop: EventLoop) -> None:
        self._loop = loop

    def result(self) -> Any:
        # TODO: support cancellation
        if self._state != "FINISHED":
            raise RuntimeError("Future should be finished")
        return self._result

    def set_result(self, result: Any) -> None:
        self._result = result
        self._state = "FINISHED"

    # def __await__(self):



class Task(Future):
    def __init__(self, coro: Coroutine, loop: EventLoop) -> None:
        super().__init__(loop)
        self._coro = coro
        self._loop.call_soon(self.__step)

    def __step(self) -> None:
        try:
            result = self._coro.send(None) # TODO: check result
        except StopIteration as exc:
            super().set_result(exc.value)
        else:
            self._loop.call_soon(self.__step)


class Handle:
    __slots__ = ("_callback", "_args", "_loop")

    def __init__(self, callback: Callable, args: Sequence[Any], loop: EventLoop) -> None:
        self._callback = callback
        self._args = args
        self._loop = loop

    def _run(self) -> None:
        # TODO: proper error handling
        self._callback(*self._args)

class TimerHandle(Handle):
    __slots__ = ("_when", "_scheduled")
    def __init__(self, when: float, callback: Callable, args: Sequence[Any], loop: EventLoop) -> None:
        super().__init__(callback, args, loop)
        self._when = when
        self._scheduled = False

    def when(self) -> float:
        return self._when


def run(coro: Coroutine):
    loop = get_event_loop()
    task = loop.create_task(coro)
    return loop.run_until_complete(task)


async def sleep(delay: float, result: Any=None) -> None:
    loop = get_event_loop()
    future = loop.create_future()
    loop.call_later(delay, future.set_result, future, result)
    return await future

