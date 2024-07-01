import socket


def main():
    host = 'httpbin.org'
    port = 80

    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Get the IP address of the host
    server_address = socket.gethostbyname(host)

    # Connect to the server
    sock.connect((server_address, port))

    # Format and send the HTTP GET request
    request = "GET /get HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"
    sock.send(request.encode('utf-8'))

    # Receive the response
    response = b""
    while True:
        data = sock.recv(4096)
        if not data:
            break
        response += data

    # Close the socket
    sock.close()

    # Print the response
    print(response.decode('utf-8'))

if __name__ == "__main__":
    main()