from collections import defaultdict
import hashlib

import httpx
import urllib
from post_service import Post
from config import Server


class ShardedPostServiceClient:
    def __init__(self, servers: list[Server]) -> None:
        self.servers = servers

    def post(self, uid: str, content: str) -> bool:
        post_server = self._shard(uid)
        post_service_url = f"http://{post_server.host}:{post_server.port}/post"
        params = {"user": uid, "content": content}
        response = httpx.post(post_service_url, params=params)
        response.raise_for_status()
        return response.json()

    def get_users_posts(
        self, uids: list[str], start_ts: float, end_ts: float
    ) -> list[Post]:
        post_servers_uids = self._shard_uids(uids)
        all_posts = []
        start_ts = urllib.parse.quote(str(start_ts))
        end_ts = urllib.parse.quote(str(end_ts))
        for post_server, uids in post_servers_uids.items():
            uids_query = urllib.parse.quote(",".join(uids))
            url = f"http://{post_server.host}:{post_server.port}/get_users_posts?uids={uids_query}&start_ts={start_ts}&end_ts={end_ts}"
            try:
                resp = httpx.get(url, timeout=3)
                if resp.status_code == 200:
                    posts = [Post(**post) for post in resp.json()]
                    all_posts.extend(posts)
                else:
                    print(
                        f"Error querying post service {post_server.host}:{post_server.port}: {resp.text}"
                    )
                    raise Exception(f"Error querying post service: {resp.status_code=}")
            except Exception as e:
                print(f"Error querying post service: {e}")
                raise e
        return all_posts

    def _shard(self, uid: str) -> Server:
        uid_hash = int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16)
        shard_index = uid_hash % len(self.servers)
        return self.servers[shard_index]

    def _shard_uids(self, uids: list[str]) -> dict[Server, list[str]]:
        uids_by_server = defaultdict(list)
        for uid in uids:
            server = self._shard(uid)
            uids_by_server[server].append(uid)
        return dict(uids_by_server)
