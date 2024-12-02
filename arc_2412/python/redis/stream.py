"""
Redis pubsub doesn't provide the classic message queue semantics. Whenever a 
publisher publishes a message, the message is sent to all the subscribers. 
"""

import argparse
import contextlib
import subprocess
import concurrent 
import redis


@contextlib.contextmanager
def start_redis_server():
    print("Starting redis server")
    proc = subprocess.Popen(["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    yield
    print("Stopping redis server")
    proc.terminate()


stream_name = "mystream"

def producer():
    with start_redis_server():
        r = redis.Redis(host="localhost", port=6379, db=0)
        print("Connected to redis server")

        while True:
            msg = input("Enter message to stream: ")
            if msg == "exit":
                break
            msg_id = r.xadd(stream_name, {"message": msg})
            print(f"Published message with id {msg_id}")

def consumer(name):
    r = redis.Redis(host="localhost", port=6379, db=0)
    print("Connected to redis server")

    try:
        r.xgroup_create(stream_name, "mygroup", id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if not str(e).startswith("BUSYGROUP Consumer Group name already exists"):
            raise

    while True:
        try:
            msgs = r.xreadgroup("mygroup", name, {stream_name: ">"}, count=1, block=10)
            for msg in msgs:
                print(f"Consumer {name} received message: {msg}")
                r.xack(stream_name, "mygroup", msg[1][0][0])
        except Exception as e:
            print(f"Consumer {name} error: {e}")

def consumers():
    N_CONSUMERS = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONSUMERS) as executor:
        futs = [executor.submit(consumer, f"consumer-{i}") for i in range(N_CONSUMERS)]
    concurrent.futures.wait(futs, return_when=concurrent.futures.ALL_COMPLETED, timeout=None)
    
    

def main():
    parser = argparse.ArgumentParser(description="Redis stream")
    parser.add_argument("--producer", action="store_true", help="Publish messages")
    parser.add_argument("--consumer", action="store_true", help="Subscribe to messages")
    args = parser.parse_args()
    if args.producer:
        producer()
    elif args.consumer:
        consumers()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
