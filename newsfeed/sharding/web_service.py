from fastapi import FastAPI
import httpx
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config import (
    SERVICE_HOST,
    WEB_SERVICE_PORT,
    WEB_SERVICE_N_WORKERS,
    FEED_SERVICE_PORT,
    post_service_servers
)
import uvicorn
from client_lib import ShardedPostServiceClient


static_dir = Path(__file__).parent / "static"

# This is the module-level app
app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def home():
    return FileResponse(static_dir / "index.html")


# Forward POST requests to the post service
@app.post("/post")
def forward_post(user: str, content: str):
    """
    This endpoint receives a post request from the client,
    then forwards it to the post service via httpx (sync).
    """
    sharded_post_service_client = ShardedPostServiceClient(post_service_servers)
    return sharded_post_service_client.post(user, content)


@app.get("/feed")
def forward_feed(user: str, start_ts: float, end_ts: float):
    """
    This endpoint receives a feed request from the client,
    then forwards it to the feed service via httpx (sync).
    """
    feed_service_url = f"http://{SERVICE_HOST}:{FEED_SERVICE_PORT}/feed"
    params = {"user": user, "start_ts": start_ts, "end_ts": end_ts}
    response = httpx.get(feed_service_url, params=params)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    uvicorn.run(
        "web_service:app",
        host=SERVICE_HOST,
        port=WEB_SERVICE_PORT,
        workers=WEB_SERVICE_N_WORKERS,
    )
