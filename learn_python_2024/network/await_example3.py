import select
import socket
import types
from collections import deque


class EventLoop:
    def __init__(self):
        self._ready = deque()
        self._read_wait = {}
        self._write_wait = {}
        self._stopped = False

    def run_forever(self):
        while not self._stopped:
            if self._ready:
                callback, args = self._ready.popleft()
                callback(*args)
            else:
                timeout = 1
                if self._read_wait or self._write_wait:
                    timeout = 0.01
                
                rlist, wlist, _ = select.select(self._read_wait.keys(), self._write_wait.keys(), [], timeout)
                for r in rlist:
                    self._ready.append(self._read_wait.pop(r))
                for w in wlist:
                    self._ready.append(self._write_wait.pop(w))

    def stop(self):
        self._stopped = True

    def call_soon(self, callback, *args):
        self._ready.append((callback, args))

    def create_task(self, coro):
        self.call_soon(self._step_coro, coro)

    def _step_coro(self, coro):
        try:
            fut = next(coro)
            fut.add_done_callback(lambda: self.call_soon(self._step_coro, coro))
        except StopIteration:
            pass

    def register_reader(self, fd, callback):
        self._read_wait[fd] = (callback, ())

    def register_writer(self, fd, callback):
        self._write_wait[fd] = (callback, ())

    def unregister_reader(self, fd):
        self._read_wait.pop(fd, None)

    def unregister_writer(self, fd):
        self._write_wait.pop(fd, None)

class Future:
    def __init__(self):
        self._done_callbacks = []

    def set_result(self, result):
        self.result = result
        for callback in self._done_callbacks:
            callback()

    def add_done_callback(self, fn):
        self._done_callbacks.append(fn)

@types.coroutine
def sleep(duration):
    fut = Future()
    loop.call_later(duration, fut.set_result, None)
    yield fut

@types.coroutine
def read(fd):
    fut = Future()
    loop.register_reader(fd, lambda: fut.set_result(None))
    yield fut
    loop.unregister_reader(fd)
    return fd.recv(1024)

@types.coroutine
def write(fd, data):
    fut = Future()
    loop.register_writer(fd, lambda: fut.set_result(None))
    yield fut
    loop.unregister_writer(fd)
    fd.send(data)

async def handle_client(client_sock):
    while True:
        data = await read(client_sock)
        if not data:
            break
        await write(client_sock, data)
    client_sock.close()

async def start_server(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    server_sock.setblocking(False)
    print(f"Server listening on {host}:{port}")

    while True:
        client_sock, addr = await accept(server_sock)
        print(f"Accepted connection from {addr}")
        loop.create_task(handle_client(client_sock))

@types.coroutine
def accept(sock):
    fut = Future()
    loop.register_reader(sock.fileno(), lambda: fut.set_result(sock.accept()))
    yield fut
    loop.unregister_reader(sock.fileno())
    return fut.result

# Create and run the event loop
loop = EventLoop()
loop.create_task(start_server('127.0.0.1', 8888))
loop.run_forever()