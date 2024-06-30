import collections
import heapq
import logging
import time
from typing import Any, Callable, Coroutine, Literal, Sequence

logger = logging.getLogger(__name__)


class EventLoop:
    def __init__(self):
        self._ready: collections.deque[Handle] = collections.deque()
        self._waiting: list[Handle] = []  # TODO: what is this used for?
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
        itr = 0
        while self._ready or self._waiting or self._scheduled:
            self._run_once()
            # TODO: support stopping
            if self._stopping:
                break
            itr += 1

    def _run_once(self) -> None:
        n_scheduled_ready = 0
        while self._scheduled and self._scheduled[0].when() <= self.time():
            timer_handle = heapq.heappop(self._scheduled)
            timer_handle._scheduled = False  # why?
            self._ready.append(timer_handle)
            n_scheduled_ready += 1
        if n_scheduled_ready:
            logger.debug(f"Moving {n_scheduled_ready} scheduled events to ready")

        while len(self._ready) > 0:
            handle = self._ready.popleft()
            # TODO: support handle is cancelled
            handle._run()

    def create_future(self) -> "Future":
        return Future(self)

    def call_later(self, delay: float, callback: Callable, *args) -> "TimerHandle":
        assert delay >= 0
        logger.debug(f"Register time event to run after {delay} seconds: {callback.__name__}()")
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
    _state: Literal["PENDING", "CANCELLED", "FINISHED"] = "PENDING"
    _result = None
    _asyncio_future_blocking = False
    # TODO: support callback

    def __init__(self, loop: EventLoop) -> None:
        self._loop = loop
        self._callbacks: list[Callable] = []

    def result(self) -> Any:
        # TODO: support cancellation
        if self._state != "FINISHED":
            raise RuntimeError("Future should be finished")
        return self._result

    def set_result(self, result: Any) -> None:
        self._result = result
        self._state = "FINISHED"
        self.__schedule_callbacks()

    def done(self) -> bool:
        # either completed, exception, or cancelled
        return self._state != "PENDING"

    def add_done_callback(self, callback: Callable) -> None:  # TODO: what is context
        if self.done():
            self._loop.call_soon(callback, self)
        else:
            self._callbacks.append(callback)

    def __schedule_callbacks(self) -> None:
        callbacks = self._callbacks[:]  # TODO: is copy needed?
        self._callbacks.clear()
        for callback in callbacks:
            self._loop.call_soon(callback, self)

    def __await__(self):
        if not self.done():
            self._asyncio_future_blocking = True  # TODO: look into what happens
            yield self
        assert self.done(), "Future should be done"
        return self.result()


class Task(Future):
    def __init__(self, coro: Coroutine, loop: EventLoop) -> None:
        super().__init__(loop)
        self._coro = coro
        self._loop.call_soon(self.__step)
        self._fut_waiter = None  # TODO: what is this used for?

    def __step(self) -> None:
        self._fut_waiter = None
        try:
            result = self._coro.send(None)  # TODO: check result
        except StopIteration as exc:
            logger.debug(f"Coro {self._coro.__name__} is done")
            super().set_result(exc.value)
            return
        else:
            blocking = getattr(result, "_asyncio_future_blocking", None)
            if blocking is not None:
                logger.debug(
                    f"Coro {self._coro.__name__} resulted in a future, blocking: {blocking}"
                )
                assert blocking is True, "Future should be blocking"
                assert result is not self, "Task shouldn't wait on itself"
                result._asyncio_future_blocking = False  # TODO: why?

                def wakeup_cb(fut):
                    logger.debug(f"Future {fut} is done. Wake up coro {self._coro.__name__}")
                    self.__wakeup(fut)

                result.add_done_callback(wakeup_cb)
                self._fut_waiter = result

            elif result is None:
                logging.info(f"Coro {self._coro.__name__} yielded None. Re-queue the coro")
                self._loop.call_soon(self.__step)
            else:
                raise RuntimeError("coro should yield None or Future")

    def __wakeup(self, future: Future) -> None:
        assert future.done(), "Future should be done"
        self.__step()


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

    def __init__(
        self, when: float, callback: Callable, args: Sequence[Any], loop: EventLoop
    ) -> None:
        super().__init__(callback, args, loop)
        self._when = when
        self._scheduled = False

    def when(self) -> float:
        return self._when


def run(coro: Coroutine):
    loop = get_event_loop()
    task = loop.create_task(coro)
    return loop.run_until_complete(task)


async def sleep(delay: float, result: Any = None) -> None:
    """
    step 1: create a future
    step 2: register a call_later event, which set_result to the future
    step 3: current coro will wait for this future

    Qs:
    - if a coro can await on a future, what does future's __await__ do?
        A:  stop existing coro (by not registering it further to the event loop)
            register a new time event, happens after the delay, which set the future's result
            yield the result to coro.step() with a flag (e.g. _asyncio_future_blocking), so that coro.step() register a callback to the future to wake up/resume the coro
    - how to make a coro waiting for sth? by setting its status as pending?
       - e.g. hwo to avoid eventloop keept polling the coro to check whether it's ready
       - likely by setting the coro's status as pending, and future compleletion will somehoe changes the status
       A: it doesn't further register the event to the eventloop, until the future is done
    - future vs task:
        - future: general awaitable object
        - task: coro which can be called with send(None)?
    """
    loop = get_event_loop()
    future = loop.create_future()

    def set_result():
        logger.debug(f"after {delay} seconds, set result to future")
        future.set_result(result)

    loop.call_later(delay, set_result)
    return await future
