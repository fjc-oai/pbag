import collections
import heapq
import logging
import selectors
import socket
import time
from typing import Any, Callable, Coroutine, Literal, Sequence

logger = logging.getLogger(__name__)


class EventLoop:
    def __init__(self):
        self._ready: collections.deque[Handle] = collections.deque()
        self._waiting: list[Handle] = []  # TODO: what is this used for?
        self._scheduled: list[TimerHandle] = []
        self._stopping = False
        self._selector = selectors.DefaultSelector() # TODO: play with selectors

    def create_task(self, coro: Coroutine) -> "Task":
        task = Task(coro, self)
        return task

    def call_soon(self, callback: Callable, *args) -> "Handle":
        handle = Handle(callback, args, self)
        self._ready.append(handle)
        return handle
    
    def stop(self):
        self._stopping = True

    def run_until_complete(self, task: "Task") -> Any:
        def stop_loop_cb(task):
            logger.info("Task is done, stop the loop")
            task.get_loop().stop()
        task.add_done_callback(stop_loop_cb)
        self.run_forever()
        return task.result()

    def run_forever(self) -> None:
        # TODO: proper error handling
        itr = 0
        # while self._ready or self._waiting or self._scheduled: # TODO: this won't work, e.g. in create_connection, there is no ready or wating events but we cannot stop the loop.
        while True:
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
            logger.info(f"Eventloop: move {n_scheduled_ready} scheduled events to ready")

        timeout = None
        if self._ready:
            timeout = 0
        # timeout == 0 means poll and return immediately
        # timeout == None means wait forever until there is an event
        n_select_reader_ready = 0
        n_select_writer_ready = 0
        event_list = self._selector.select(timeout) # TODO: this is the trick!!!
        for key, mask in event_list:
            fileobj, (reader, writer) = key.fileobj, key.data
            if mask & selectors.EVENT_READ and reader is not None:
                self._ready.append(reader)
                n_select_reader_ready += 1
            if mask & selectors.EVENT_WRITE and writer is not None:
                self._ready.append(writer)
                n_select_writer_ready += 1

        if n_select_reader_ready or n_select_writer_ready:
            logger.info(f"Eventloop: move {n_select_reader_ready} reader and {n_select_writer_ready} writer events to ready")

        logger.debug(f"Eventloop: run {len(self._ready)} ready events")
        while len(self._ready) > 0:
            handle = self._ready.popleft()
            # TODO: support handle is cancelled
            handle._run()

    def create_future(self) -> "Future":
        return Future(self)

    def call_later(self, delay: float, callback: Callable, *args) -> "TimerHandle":
        assert delay >= 0
        logger.info(f"Register time event to run after {delay} seconds: {callback.__name__}()")
        return self.call_at(self.time() + delay, callback, *args)

    def call_at(self, when: float, callback: Callable, *args) -> "TimerHandle":
        timer = TimerHandle(when, callback, args, self)
        heapq.heappush(self._scheduled, timer)
        timer._scheduled = True
        return timer

    def time(self):
        return time.monotonic()

    def _add_writer(self, fd, callback, *args):
        handle = Handle(callback, args, self)
        key = self._selector.get_map().get(fd)
        assert key is None, f"fd {fd} is already registered"
        self._selector.register(fd, selectors.EVENT_WRITE, (None, handle))
        return handle

    def _sock_connect_cb(self, fut, sock, address):
        assert not fut.done(), "Future shouldn't be done"
        try:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)  # TODO: what is this?
            if err != 0:
                raise OSError(err, f"Connect call failed {address}")
        except (BlockingIOError, InterruptedError):
            pass
        except BaseException as exc:
            fut.set_exception(exc)  # TODO: implement this
        else:
            logger.info(f"Socket {sock} connected successfully")
            fut.set_result(None)

    async def create_connection(self, host: str, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        fd = sock.fileno()
        fut = self.create_future()
        try:
            sock.connect((host, port))
        except (BlockingIOError, InterruptedError):  # TODO: what are these exceptions?
            handle = self._add_writer(fd, self._sock_connect_cb, fut, sock, (host, port))
            # TODO: do we need to do fut.add_done_callback(self._sock_write_done) ?
        else:
            assert False, "Shouldn't reach here"
            fut.set_result(None)
        await fut # TODO: make fut.set_result(sock)
        return sock
    
    async def sock_sendall(self, sock: socket.socket, data: bytes|str) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        assert sock.gettimeout() == 0, "Socket must be non-blocking"
        try: 
            n = sock.send(data)
        except (BlockingIOError, InterruptedError): # TODO: look into details
            n = 0
        
        # TODO: should we enforce this?
        # if n == len(data):
        #     assert False, "Shouldn't reach here"
        
        # TODO: implement iteratively sending the remaining data
        fut = self.create_future()
        fd = sock.fileno()
        # TODO: does the same sock (e.g. connect, write) return the same fd?
        def write_cb(fut, sock):
            logger.info(f"Socket {sock} write done")
            fut.set_result(len(data))
        handle = self._add_writer(fd, write_cb, fut, sock)
        # TODO: do we need to do fut.add_done_callback(self._sock_write_done) ?
        return await fut



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
    
    def get_loop(self):
        return self._loop


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
            logger.info(f"Coro {self._coro.__name__} is done")
            super().set_result(exc.value)
            return
        else:
            blocking = getattr(result, "_asyncio_future_blocking", None)
            if blocking is not None:
                logger.info(
                    f"Coro {self._coro.__name__} resulted in a future, blocking: {blocking}"
                )
                assert blocking is True, "Future should be blocking"
                assert result is not self, "Task shouldn't wait on itself"
                result._asyncio_future_blocking = False  # TODO: why?

                def wakeup_cb(fut):
                    logger.info(f"Future {fut} is done. Wake up coro {self._coro.__name__}")
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

    - eventloop.run_until_complete(task) uses task's completion to stop the eventloop. before it stops, it keeps polling the eventloop to run the ready events
        does it mean the eventloop is busy looping until the task is done? thus will take 100% CPU? assuming select timeout is 0, and there is only one sleep event scheduled.
    """
    loop = get_event_loop()
    future = loop.create_future()

    def set_result():
        logger.debug(f"after {delay} seconds, set result to future")
        future.set_result(result)

    loop.call_later(delay, set_result)
    return await future

async def create_connection(host: str, port: int):
    loop = get_event_loop()
    sock = await loop.create_connection(host, port)
    return sock

async def sock_sendall(sock: socket.socket, data: bytes|str) -> None:
    loop = get_event_loop()
    return await loop.sock_sendall(sock, data)