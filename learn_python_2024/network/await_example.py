import select
import socket
from collections import deque


class SimpleEventLoop:
    def __init__(self):
        self.ready = deque()  # Queue of tasks ready to run
        self.read_waiting = {}  # Dictionary of tasks waiting for read
        self.write_waiting = {}  # Dictionary of tasks waiting for write

    def create_task(self, coro):
        print("Creating task to ready queue")
        self.ready.append(Task(coro))

    def run_until_complete(self, coro):
        self.create_task(coro)
        while self.ready or self.read_waiting or self.write_waiting:
            if self.ready:
                print("Running task")
                task = self.ready.popleft()
                try:
                    event_type, fd = task.step()
                    if event_type == 'read':
                        self.read_waiting[fd] = task
                    elif event_type == 'write':
                        self.write_waiting[fd] = task
                except StopIteration as e:
                    pass
            else:
                print("Waiting for I/O")
                self._select()

    def _select(self):
        timeout = None
        if self.read_waiting or self.write_waiting:
            rlist, wlist, _ = select.select(self.read_waiting.keys(), self.write_waiting.keys(), [], timeout)
            for fd in rlist:
                task = self.read_waiting.pop(fd, None)
                if task:
                    self.ready.append(task)
            for fd in wlist:
                task = self.write_waiting.pop(fd, None)
                if task:
                    print("Adding task to ready queue")
                    self.ready.append(task)

class Task:
    def __init__(self, coro):
        self.coro = coro

    def step(self):
        try:
            return self.coro.send(None)
        except StopIteration:
            return None

class AwaitableSocket:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)

    async def connect(self):
        try:
            self.sock.connect_ex((self.host, self.port))
        except BlockingIOError:
            pass

        await self._await_writable()
        print("Socket is connected")

    async def send(self, data):
        await self._await_writable()
        self.sock.sendall(data)

    async def recv(self, bufsize):
        await self._await_readable()
        return self.sock.recv(bufsize)

    async def _await_readable(self):
        while True:
            readable, _, _ = select.select([self.sock], [], [], 0.1)
            if self.sock in readable:
                break

    async def _await_writable(self):
        while True:
            _, writable, _ = select.select([], [self.sock], [], 0.1)
            if self.sock in writable:
                break

    def close(self):
        self.sock.close()


async def fetch_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    awaitable_sock = AwaitableSocket('127.0.0.01', 12345)

    await awaitable_sock.connect()  # Connect to server
    print("Connected to server")

    # request = b"GET / HTTP/1.0\r\nHost: example.com\r\n\r\n"
    request = "hallo".encode('utf-8')
    await awaitable_sock.send(request)  # Send HTTP request
    print("Request sent")

    response = await awaitable_sock.recv(4096)  # Read response
    print(f"Received: {response.decode('utf-8')}")

event_loop = SimpleEventLoop()
event_loop.run_until_complete(fetch_data())