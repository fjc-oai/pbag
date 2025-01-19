from pathlib import Path

import uvicorn
from config import SERVICE_HOST, WEB_SERVICE_PORT
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator


def web_service_handler() -> FastAPI:
    app = FastAPI()
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def home():
        return FileResponse(static_dir / "index.html")

    return app


def create_web_service() -> None:
    print(f"Starting web service on {SERVICE_HOST}:{WEB_SERVICE_PORT}")
    uvicorn.run(web_service_handler(), host=SERVICE_HOST, port=WEB_SERVICE_PORT)


if __name__ == "__main__":
    create_web_service()
