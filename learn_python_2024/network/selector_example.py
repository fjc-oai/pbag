"""
Non-blocking server-client communication using selector module. Handle iteratively recv and send data.
Written with the help from ChatGPT.

To run the code:

    > python select_example.py server
    Server is listening on 127.0.0.1:12345
    server_socket is non-blocking
    Connected by ('127.0.0.1', 53778). Conn is non-blocking
    Received from ('127.0.0.1', 53778): python is the best language because

    Connection closed by ('127.0.0.1', 53778)

    
    > python selector_example.py client
    Connected to localhost:12345
    Sent: python is the best language because

    Received: Server ECHO: python is the best language because


Major APIs:
    - selector
        - register() args
        - select() return value
        - select() arg timeout
    - nonblocking socket
        - abstraction: only read/write is provided, no ordering, nor which event correspond to which read or write
        - read(): 
            - iterative read over the buffer  
            - handle when data is smaller then the size
            - handle when data is larger then the size
        - write()
            - iterative write over the buffer, each time the return value is the number of bytes written  

    class selectors.SelectorKey
        A SelectorKey is a namedtuple used to associate a file object to its underlying file descriptor, selected event mask and attached data. It is returned by several BaseSelector methods.

        fileobj: File object registered.
        fd: Underlying file descriptor.
        events: Events that must be waited for on this file object.
        data: Optional opaque data associated to this file object: for example, this could be used to store a per-client session ID.

    abstractmethod register(fileobj, events, data=None)
"""

import argparse
import selectors
import socket

HOST = "localhost"
PORT = 12345


def server():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, PORT))
    server_sock.listen()
    server_sock.setblocking(False)

    # TODO: support multiple clients
    read_buffer = ""
    write_buffer = ""

    def write_cb(conn):
        nonlocal write_buffer
        bytes_sent = conn.send(write_buffer.encode())
        print(f"Sent {bytes_sent} bytes: {write_buffer[:bytes_sent]}")
        write_buffer = write_buffer[bytes_sent:]
        if write_buffer:
            print("Waiting for more data...")
        else:
            conn.shutdown(socket.SHUT_WR)
            print("Data sent successfully. Now unregistering the write event.")
            selector.unregister(conn)

    def read_cb(conn):
        nonlocal read_buffer
        nonlocal write_buffer
        data = conn.recv(8)
        if data and "\r\n" not in data.decode():
            partial_message = data.decode()
            print(f"Partial message received: {partial_message}")
            print("Waiting for more data...")
            read_buffer += partial_message
        else:  # data is empty or "\r\n" is in data
            print(f"Full message received: {read_buffer}")
            write_buffer = read_buffer.upper()
            read_buffer = ""
            print(f"Now sending response: {write_buffer}")
            selector.modify(conn, selectors.EVENT_WRITE, write_cb)

    def accept_cb(server_sock):
        conn, addr = server_sock.accept()
        print(f"Accepted connection from {addr}")
        conn.setblocking(False)
        selector.register(conn, selectors.EVENT_READ, read_cb)

    selector = selectors.DefaultSelector()
    selector.register(server_sock, selectors.EVENT_READ, accept_cb)
    print(f"Server is listening on {HOST}:{PORT}")

    try:
        while True:
            # Timeout:
            #   0 means non-blocking
            #   > 0 means blocking for that amount of time
            #   None means infinite blocking
            events = selector.select(timeout=None)
            for key, mask in events:
                callback = key.data
                callback(key.fileobj)
    except KeyboardInterrupt:
        print("Server is shutting down")
    finally:
        server_sock.close()
        selector.close()


def client():
    # Run in blocking mode for simplicity
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"Connected to {HOST}:{PORT}")
    message = "python is the best language because \n"
    sock.sendall(message.encode())
    print(f"Sent: {message}")
    data = sock.recv(1024)
    print(f"Received: {data.decode()}")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("role", choices=["server", "client"])
    args = argparser.parse_args()
    if args.role == "server":
        server()
    else:
        client()


if __name__ == "__main__":
    main()
