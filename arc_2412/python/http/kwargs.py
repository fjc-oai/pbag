import argparse
from fastapi import FastAPI
import uvicorn
import time
import httpx


class Core:
    def compute(self, a: int, b: int, c: int) -> int:
        print(f"Computing: {a=}, {b=}, {c=}")
        return (a + b) * c


def service_handler(core: Core) -> FastAPI:
    app = FastAPI()

    @app.post("/compute")
    async def compute(data: dict):
        print(f"Received data: {data}")
        res = core.compute(**data)
        # res = foo(**data)
        return {"data": res}

    return app


def create_service():
    core = Core()
    HOST = "0.0.0.0"
    PORT = 8001
    uvicorn.run(service_handler(core), host=HOST, port=PORT)


def server():
    create_service()
    while True:
        print("Server is running...")
        time.sleep(30)


def client():
    url = "http://localhost:8001/compute"
    client = httpx.Client()
    req = {"a": 1, "b": 2, "c": 3}
    try:
        resp = client.post(url, json=req)
        print(resp.json())
    except Exception as e:
        print(f"{e=}")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["server", "client"])
    args = argparser.parse_args()
    if args.mode == "server":
        server()
    elif args.mode == "client":
        client()


if __name__ == "__main__":
    main()
