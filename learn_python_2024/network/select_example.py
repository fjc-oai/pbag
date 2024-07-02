"""
Non-blocking server-client communication using select module.
Written with the help from ChatGPT.

To run the code:

    > python select_example.py server
    Server is listening on 127.0.0.1:12345
    server_socket is non-blocking
    Connected by ('127.0.0.1', 53662). Conn is non-blocking
    Received from ('127.0.0.1', 53662): Client 9: Hi there!
    Connection closed by ('127.0.0.1', 53662)

    > python select_example.py client
    Received: Server ECHO: Client 9: Hi there!
"""
import argparse
import random
import select
import socket

HOST = "127.0.0.1"
PORT = 12345


def server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    # Socket is blocking by default. Use getblocking() and setblocking() to
    # check and set the blocking status.
    server_socket.setblocking(False)
    print(f"Server is listening on {HOST}:{PORT}")
    print(f"server_socket is {'blocking' if server_socket.getblocking() else 'non-blocking'}")


    socket_list = [server_socket]
    clients = {}

    while True:
        """
        The select.select() function in Python takes three mandatory arguments
        and one optional argument:
            1.  rlist: A list of file descriptors to be checked for readability.
            2.  wlist: A list of file descriptors to be checked for writability.
            3.  xlist: A list of file descriptors to be checked for exceptional
                conditions (e.g., errors).
            4.  timeout (optional): A timeout value in seconds. This can be a
                floating-point number for subsecond precision. If omitted or
                None, select can block indefinitely until at least one file
                descriptor is ready.
        Returns:
            1.  Readable: A list of file descriptors that are ready for reading.
	        2.  Writable: A list of file descriptors that are ready for writing.
	        3.  Exceptional: A list of file descriptors that have an exceptional
	            condition (usually errors).

        Blocking vs Non-blocking:
            The select() is a blocking function by default. It will block until
            at least one file descriptor is ready. Setting timeout=0 makes it a
            non-blocking function. It will return immediately even if no file
            descriptor is ready.
        """
        read_sockets, _, _ = select.select(socket_list, [], [])
        for sock in read_sockets:
            if sock == server_socket:
                conn, addr = server_socket.accept()
                conn.setblocking(False)
                socket_list.append(conn)
                clients[conn] = addr
                print(f"Connected by {addr}. Conn is {'blocking' if conn.getblocking() else 'non-blocking'}")

            else:
                """
                The recv() receives up to 1024 bytes of data from the socket. If
                the data is larger than 1024 bytes, it does not guarantee that
                the entire data is received in one call. It's the caller's
                responsibility to call recv() multiple times to receive the
                entire data, or use a higher-level protocol to handle the data.
                """
                data = sock.recv(1024)
                if data:
                    print(f"Received from {clients[sock]}: {data.decode()}")
                    reply = f"Server ECHO: {data.decode()}"
                    """
                    The send() function sends data to the socket. It returns the
                    number of bytes sent. It may not send the entire data in one
                    call. The caller should call send() multiple times to send
                    the entire data. The blocking behavior of send() depends on
                    the socket's blocking status.

                    The sendall() function sends the entire data to the socket.
                    It will block until all data is sent. If the socket is in
                    non-blocking mode, it will raise BlockingIOError if the
                    operation would block.
                    """
                    sock.send(reply.encode())
                else:
                    sock.close()
                    socket_list.remove(sock)
                    print(f"Connection closed by {clients[sock]}")
                    del clients[sock]


def client(id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    msg = f"Client {id}: Hi there!"
    client_socket.sendall(msg.encode())
    data = client_socket.recv(1024)
    print(f"Received: {data.decode()}")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("role", choices=["server", "client"])
    arg = argparser.parse_args()

    if arg.role == "server":
        server()
    elif arg.role == "client":
        client(random.randint(1, 10))
    else:
        argparser.print_help()


if __name__ == "__main__":
    main()
