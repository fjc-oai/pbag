"""
Blocking socket example for server-client communication.
"""

import argparse
import socket

HOST = "127.0.0.1"
PORT = 12345

def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        """
        After the server is bound to an address and port, the server listens for incoming connections.
        To inspect:
            > netstat -an | grep LISTEN
            > lsof -i -P -n | grep LISTEN
            > nc -zv localhost 10000-20000 2>&1 | grep succeeded
                Connection to localhost port 12345 [tcp/italk] succeeded!
        To connect from terminal:
            > nc localhost 12345
            >>> Hello, World

        """
        while True:
            print("Waiting for connection...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if data:
                        print(f"Received: {data.decode()}")
                    else:
                        break
                    conn.sendall(data)

def client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"Hello, World")
        data = s.recv(1024)
        print(f"Received: {data.decode()}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("role", choices=["server", "client"])
    args = argparser.parse_args()
    if args.role == "server":
        server()
    elif args.role == "client":
        client()
    else:
        argparser.print_help()