import socket

from debugee import HOST, PORT

def send_command(sock, command):
    print(f"sending command: {command}")
    sock.sendall(command.encode('utf-8') + b'\n')
    try:
        while True:
            res = sock.recv(1024)
            print(res.decode('utf-8'))
    except: 
        pass

def debugger():
    with socket.create_connection((HOST, PORT)) as sock:
        sock.setblocking(False)
        while True:
            command = input(">>> ")
            send_command(sock, command)

if __name__ == '__main__':
    debugger()