import random
import urllib

import httpx
import uvicorn
from config import (
    FEED_SERVICE_PORT,
    POST_SERVICE_PORT,
    SERVICE_HOST,
    FEED_BURN_CPU_MS,
)
from fastapi import FastAPI, requests
from fastapi.middleware.cors import CORSMiddleware
from post_service import Post
from utils import create_users, validate_users, burn_cpu
from prometheus_fastapi_instrumentator import Instrumentator


class FeedService:
    def __init__(self, users: dict[str, set[str]]) -> None:
        self.users: dict[str, set[str]] = users

    def feed(self, uid: str, start_ts: float, end_ts: float) -> list[Post]:
        # Get the list of users that the user follows
        assert uid in self.users, f"User {uid} does not exist"
        friends = self.users.get(uid, set())
        posts = self.query_post_service([uid] + list(friends), start_ts, end_ts)
        if posts is None:
            return []
        posts.sort(key=lambda post: post.timestamp)
        burn_cpu(FEED_BURN_CPU_MS)
        return posts

    def query_post_service(
        self, uids: list[str], start_ts: float, end_ts: float
    ) -> list[Post]:
        uids_query = urllib.parse.quote(",".join(uids))
        start_ts = urllib.parse.quote(str(start_ts))
        end_ts = urllib.parse.quote(str(end_ts))
        url = f"http://{SERVICE_HOST}:{POST_SERVICE_PORT}/get_users_posts?uids={uids_query}&start_ts={start_ts}&end_ts={end_ts}"
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                posts = [Post(**post) for post in resp.json()]
                return posts
            else:
                print(f"Error querying post service: {resp.text}")
                raise Exception(f"Error querying post service: {resp.status_code=}")
        except Exception as e:
            print(f"Error querying post service: {e}")
            raise e


def feed_service_handler(feed_service: FeedService) -> FastAPI:
    app = FastAPI()
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # Allow CORS. Otherwise, the frontend cannot properly render the feed !!!!!! IMPORTANT !!!!!!
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # or a list of specific origins/domains if you prefer
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/feed")
    def feed(user: str, start_ts: float, end_ts: float):
        return feed_service.feed(user, start_ts, end_ts)

    return app


def create_feed_service() -> None:
    print(f"Starting feed service on {SERVICE_HOST}:{FEED_SERVICE_PORT}")
    users = create_users()
    validate_users(users)
    feed_service = FeedService(users)

    uvicorn.run(
        feed_service_handler(feed_service), host=SERVICE_HOST, port=FEED_SERVICE_PORT
    )


if __name__ == "__main__":
    create_feed_service()
