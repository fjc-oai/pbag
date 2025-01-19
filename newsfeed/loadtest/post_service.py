import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import uvicorn
from config import POST_SERVICE_PORT, SERVICE_HOST
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
        self.user_posts: dict[str, list[str]] = defaultdict(list)
        self.posts: dict[str, Post] = {}
        self.lock = threading.Lock()

    def post(self, uid: str, content: str) -> bool:
        timestamp = float(time.time())
        post_id = f"{uid}-{timestamp}"
        post = Post(uid, post_id, content, timestamp)
        with self.lock:
            self.posts[post_id] = post
            self.user_posts[uid].append(post_id)
        return True

    def get_users_posts(self, uids: list[str], start_ts: float, end_ts: float) -> list[Post]:
        post_ids = []
        with self.lock:
            for uid in uids:
                post_ids += [
                    post_id
                    for post_id in self.user_posts[uid]
                    if start_ts <= self.posts[post_id].timestamp <= end_ts
                ]
            posts = [self.posts[post_id] for post_id in post_ids]
        return posts


def post_service_handler(post_service: PostService) -> FastAPI:
    app = FastAPI()
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/post")
    def post(user: str, content: str):
        print(f"User {user} is posting {content}")
        return post_service.post(user, content)

    @app.get("/get_users_posts")
    def get_users_posts(
        uids: str, start_ts: float, end_ts: float
    ):  # TODO (mhr): use list[str] instead of str hack!!!
        uids = uids.split(",")
        return post_service.get_users_posts(uids, start_ts, end_ts)

    return app


def create_post_service() -> None:
    print(f"Starting post service on {SERVICE_HOST}:{POST_SERVICE_PORT}")
    post_service = PostService()
    uvicorn.run(post_service_handler(post_service), host=SERVICE_HOST, port=POST_SERVICE_PORT)


if __name__ == "__main__":
    create_post_service()
