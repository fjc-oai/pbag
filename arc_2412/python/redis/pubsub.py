"""
Redis pubsub doesn't provide the classic message queue semantics. Whenever a 
publisher publishes a message, the message is sent to all the subscribers. 
"""

import argparse
import contextlib
import subprocess

import redis


@contextlib.contextmanager
def start_redis_server():
    print("Starting redis server")
    proc = subprocess.Popen(["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    yield
    print("Stopping redis server")
    proc.terminate()


def pub():
    with start_redis_server():
        r = redis.Redis(host="localhost", port=6379, db=0)
        print("Connected to redis server")

        pubsub = r.pubsub()
        pubsub.subscribe("mychannel")
        print("Subscribed to mychannel")

        while True:
            msg = input("Enter message to publish: ")
            if msg == "exit":
                break
            r.publish("mychannel", msg)


def sub():
    r = redis.Redis(host="localhost", port=6379, db=0)
    print("Connected to redis server")
    pubsub = r.pubsub()
    pubsub.subscribe("mychannel")
    print("Subscribed to mychannel")
    for msg in pubsub.listen():
        if msg["type"] == "message":
            print(msg["data"].decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Redis pubsub")
    parser.add_argument("--pub", action="store_true", help="Publish messages")
    parser.add_argument("--sub", action="store_true", help="Subscribe to messages")
    args = parser.parse_args()
    if args.pub:
        pub()
    elif args.sub:
        sub()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
