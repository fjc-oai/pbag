from fastapi import FastAPI


class PostService:
    def __init__(self) -> None:
        pass

    def post(self, uid: str, message: str) -> bool:
        pass

    def get_post_ids(self, uid: str, start_ts: float, end_ts: float) -> list[str]:
        pass


def post_service_handler(post_service: PostService) -> FastAPI:
    app = FastAPI()

    @app.post("/post")
    def post(user: str, content: str):
        print(f"User {user} is posting {content}")
        return post_service.post(user, content)

    # TODO (hairong) implement the get_post_ids endpoint

    return app

# TODO (hairong) implement the create_post_service function

# TODO (hairong) implement the main function
