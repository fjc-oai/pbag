import uvicorn
from config import WEB_SERVICE_HOST, WEB_SERVICE_PORT
from fastapi import FastAPI
from utils import create_users, validate_users

from newsfeed import Newsfeed


def web_service_handler(newsfeed: NewsFeed) -> FastAPI:
    app = FastAPI()

    @app.get("/post")
    def post(user: str, message: str):
        return newsfeed.post(user, message)

    @app.get("/feed")
    def feed(user: str, start_ts: float, end_ts: float):
        return newsfeed.feed(user, start_ts, end_ts)

    @app.get("/home")
    def home():
        return "Welcome to the Newsfeed!"

    return app


def create_web_service() -> None:
    users = create_users()
    validate_users(users)
    nf = NewsFeed(users)

    print(f"Starting web service on {WEB_SERVICE_HOST}:{WEB_SERVICE_PORT}")
    uvicorn.run(web_service_handler(nf), host=WEB_SERVICE_HOST, port=WEB_SERVICE_PORT)


def main() -> None:
    create_web_service()


if __name__ == "__main__":
    main()
