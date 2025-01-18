class Newsfeed:
    def __init__(self, users: dict[str, set[str]]) -> None:
        pass

    def post(self, uid: str, message: str) -> bool:
        pass

    def feed(self, uid: str, start_ts: int, end_ts: int) -> list[str]:
        pass
