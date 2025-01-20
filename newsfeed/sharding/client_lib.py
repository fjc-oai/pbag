from dataclasses import dataclass

from post_service import Post


@dataclass
class Server:
    host: str
    port: int

class ShardedPostServiceClient:
    def __init__(self, servers: list[Server]) -> None:
        pass

    def post(self, uid: str, content: str) -> bool:
        pass

    def get_users_posts(self, uids: list[str], start_ts: float, end_ts: float) -> list[Post]:
        pass

    def _shard(self, uid: str) -> Server:
        pass

    def _shard_uids(self, uids: list[str]) -> dict[Server, list[str]]:
        pass
