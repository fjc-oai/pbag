"""
A push-pull model client server communication, upon a distributed queue, using
Redis.

Client and server communicate using a Redis queue. The client submits tasks to
the queue and waits for the server to process them. The server processes the
tasks and submits the responses to the hash.

Server may have random errors while processing the tasks. The client retries
the tasks if it doesn't receive a response within a timeout.

https://redis.io/docs/data-types/sorted-sets/ 
"""

import argparse
import dataclasses
import json
import random
import subprocess
import time
from typing import Type, TypeVar

import redis

T = TypeVar("T", bound="Serializable")


class Serializable:
    def serialize(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode("utf-8")

    @staticmethod
    def deserialize(cls: Type[T], data: bytes) -> T:
        return cls(**json.loads(data.decode("utf-8")))


@dataclasses.dataclass
class Request(Serializable):
    id: str
    payload: str


@dataclasses.dataclass
class Response(Serializable):
    id: str
    payload: str


class RedisQueue:
    TASK_QUEUE: str = "task_queue"
    RESP_HASH: str = "resp_hash"

    def __init__(self, create=False):
        self._create = create
        self._redis_proc = None
        self._redis_client = None

    def _start_redis_server(self):
        print("Starting redis server")
        self._redis_proc = subprocess.Popen(
            ["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def _stop_redis_server(self):
        print("Stopping redis server")
        self._redis_proc.terminate()

    def _connect_to_redis(self):
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        while True:
            try:
                self._redis_client.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(1)
                pass
        print("Connected to redis server")

    def __enter__(self):
        if self._create:
            self._start_redis_server()
        self._connect_to_redis()
        if self._create:
            self._redis_client.flushall()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._create:
            self._stop_redis_server()

    def submit_task(self, task: bytes) -> None:
        self._redis_client.rpush(self.TASK_QUEUE, task)

    def get_task(self, timeout: None | float) -> bytes:
        return self._redis_client.blpop(self.TASK_QUEUE, timeout=timeout)[1]

    def submit_resp(self, task_id: str, resp: bytes) -> None:
        self._redis_client.hset(self.RESP_HASH, task_id, resp)

    def get_resp(self, task_id: str) -> bytes:
        resp = self._redis_client.hget(self.RESP_HASH, task_id)
        if resp:
            self._redis_client.hdel(self.RESP_HASH, task_id)
        return resp


class Client:
    def __init__(self):
        self._queue = RedisQueue(create=True)

    def __enter__(self):
        self._queue.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._queue.__exit__(exc_type, exc_value, traceback)

    def query(self, req: Request) -> Response:
        serialized_req = req.serialize()
        self._queue.submit_task(serialized_req)
        while True:
            resp = self._queue.get_resp(req.id)
            if resp:
                deserialized_resp = Response.deserialize(Response, resp)
                return deserialized_resp
            time.sleep(3)
            print(f"Waiting for response from server for task {req.id}")

    def query_with_retry(self, req: Request) -> Response:
        serialized_req = req.serialize()
        self._queue.submit_task(serialized_req)
        N_RETRY = 3
        TIMEOUT = 10
        for _ in range(N_RETRY):
            st = time.time()
            while True:
                print(f"Waiting for response from server for task {req.id}")
                resp = self._queue.get_resp(req.id)
                if resp:
                    deserialized_resp = Response.deserialize(Response, resp)
                    return deserialized_resp
                if time.time() - st > TIMEOUT:
                    break
                time.sleep(3)
            print(f"Retrying task {req.id}...")
            self._queue.submit_task(serialized_req)


class Server:
    def __init__(self) -> None:
        self._queue = RedisQueue(create=False)

    def __enter__(self):
        self._queue.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._queue.__exit__(exc_type, exc_value, traceback)

    def run(self) -> None:
        while True:
            print("Pulling task...")
            task = self._queue.get_task(timeout=None)
            if task:
                deserialized_task = Request.deserialize(Request, task)
                try:
                    resp = self.process_task(deserialized_task)
                except Exception as e:
                    print(f"Error processing task {deserialized_task.id}: {e}")
                    continue
                else:
                    serialized_resp = resp.serialize()
                    self._queue.submit_resp(deserialized_task.id, serialized_resp)

    def process_task(self, req: Request) -> Response:
        if random.random() < 0.5:
            raise Exception("Random error")
        print(f"Processing task {req.id} with payload {req.payload}")
        time.sleep(3)
        print(f"Submitting response for task {req.id}")
        return Response(req.id, req.payload.upper())


def client():
    with Client() as client:
        payloads = ["hello", "world", "foo", "bar"]
        for id, payload in enumerate(payloads, start=100):
            req = Request(id, payload)
            print(f"Sending request {req.id} with payload {req.payload}")
            resp = client.query_with_retry(req)
            print(f"Received response {resp.id} with payload {resp.payload}")
    print("Done!")


def server():
    with Server() as server:
        server.run()


def main():
    parser = argparse.ArgumentParser(description="Redis distributed queue")
    parser.add_argument("--client", action="store_true", help="Run client")
    parser.add_argument("--server", action="store_true", help="Run server")
    args = parser.parse_args()

    if args.client:
        client()
    elif args.server:
        server()


if __name__ == "__main__":
    main()
