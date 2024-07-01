import argparse
import asyncio
import socket

import aiohttp
import requests


def http_get_socket():
    host = 'httpbin.org'
    port = 80
    request = "GET /get HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Get the IP address of the host
    remote_ip = socket.gethostbyname(host)
    
    # Connect to the server
    sock.connect((remote_ip, port))
    
    try:
        # Send the HTTP GET request
        sock.sendall(request.encode('utf-8'))
        
        # Receive the response
        response = ""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data.decode('utf-8')
    
    finally:
        # Close the socket
        sock.close()

    print(response)    

def http_get_request():
    url = 'http://httpbin.org/get'
    response = requests.get(url)
    print(response.text)


async def _fetch_data():
    url = 'http://httpbin.org/get'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
        
        
def http_get_aiohttp():
    response = asyncio.run(_fetch_data())
    print(response)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()   
    argparser.add_argument('method', type=str, default='socket')
    args = argparser.parse_args()

    if args.method == 'socket':
        http_get_socket()
    elif args.method == 'requests':
        http_get_request()
    elif args.method == 'aiohttp':
        http_get_aiohttp()
    else:
        raise ValueError(f"Invalid method: {args.method}")
    
    