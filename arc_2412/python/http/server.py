from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
from urllib.parse import urlparse, parse_qs


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Printing client address on receiving a request
        client_ip, client_port = self.client_address
        params = parse_qs(urlparse(self.path).query)
        logger.info(f"Received a GET request from {client_ip}:{client_port} with {params=}")

        # Your existing response handling code
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><body><h1>Hello, World!</h1></body></html>", "utf-8"))

    def do_POST(self):
        # Printing client address on receiving a request
        client_ip, client_port = self.client_address
        params = parse_qs(urlparse(self.path).query)
        logger.info(f"Received a POST request from {client_ip}:{client_port} with {params=}")

        # Your existing POST handling code
        # ...


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ("", 8000)
    httpd = server_class(server_address, handler_class)
    print("Server started at localhost:8000")
    httpd.serve_forever()


run()


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ("", 8000)  # Host on all network interfaces, port 8000
    httpd = server_class(server_address, handler_class)
    print("Server started at localhost:8000")
    httpd.serve_forever()


def main():
    run()


if __name__ == "__main__":
    main()
