import threading
import time
from collections import defaultdict
from dataclasses import dataclass
import uvicorn

from config import POST_SERVICE_PORT, SERVICE_HOST, POST_SERVICE_N_WORKERS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


@dataclass
class Post:
    uid: str
    post_id: str
    content: str
    timestamp: float


class PostService:
    def __init__(self) -> None:
        print("Initializing PostService")
        self.user_posts: dict[str, list[str]] = defaultdict(list)
        self.posts: dict[str, Post] = {}
        self.lock = threading.Lock()

    def post(self, uid: str, content: str) -> bool:
        timestamp = float(time.time())
        post_id = f"{uid}-{timestamp}"
        new_post = Post(uid, post_id, content, timestamp)
        with self.lock:
            self.posts[post_id] = new_post
            self.user_posts[uid].append(post_id)
        return True

    def get_users_posts(self, uids: list[str], start_ts: float, end_ts: float) -> list[Post]:
        post_ids = []
        with self.lock:
            for uid in uids:
                # gather all post_ids within the time range
                post_ids += [
                    pid
                    for pid in self.user_posts[uid]
                    if start_ts <= self.posts[pid].timestamp <= end_ts
                ]
        return [self.posts[pid] for pid in post_ids]


def create_app() -> FastAPI:
    # Create a new FastAPI app
    app = FastAPI()

    # Instrument the app for Prometheus
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # Add CORS if needed
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize PostService at startup (one instance per worker)
    @app.on_event("startup")
    def startup_event():
        # Store the PostService on app.state
        app.state.post_service = PostService()

    # Define endpoints, accessing the service from app.state
    @app.post("/post")
    def post_endpoint(user: str, content: str):
        print(f"User {user} is posting {content}")
        return app.state.post_service.post(user, content)

    @app.get("/get_users_posts")
    def get_users_posts_endpoint(uids: str, start_ts: float, end_ts: float):
        uid_list = uids.split(",")
        return app.state.post_service.get_users_posts(uid_list, start_ts, end_ts)

    return app


# Create a module-level 'app' so Uvicorn can import it via "post_service:app"
app = create_app()

if __name__ == "__main__":
    print(f"Starting Post Service on {SERVICE_HOST}:{POST_SERVICE_PORT}")
    uvicorn.run(
        "post_service:app",  # Import string
        host=SERVICE_HOST,
        port=POST_SERVICE_PORT,
        workers=POST_SERVICE_N_WORKERS,
        reload=False
    )