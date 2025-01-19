from collections import defaultdict
from dataclasses import dataclass
import time

@dataclass
class Post:
    uid: str
    post_id: str
    content: str
    timestamp: float

class Newsfeed:
    def __init__(self, users: dict[str, set[str]]) -> None:
        self.users = users
        self.user_posts: dict[str, list[str]] = defaultdict(list)
        self.posts: dict[str, Post] = {}


    def post(self, uid: str, message: str) -> bool:
        timestamp = float(time.time())
        post_id = f"{uid}-{timestamp}"
        post = Post(uid, post_id, message, timestamp)
        self.posts[post_id] = post
        self.user_posts[uid].append(post_id)
        return True


    def feed(self, uid: str, start_ts: float, end_ts: float) -> list[Post]:
        post_ids = []
    
        # Get the list of users that the user follows
        assert uid in self.users, f"User {uid} does not exist"
        friends = self.users.get(uid, set())
    
        # Get the list of posts that the user has made between the start and end timestamps
        post_ids += self._get_post_ids(uid, start_ts, end_ts)
 
        # Get the list of posts that the user follows
        for user in friends:
            post_ids += self._get_post_ids(user, start_ts, end_ts)

        # Return the list of posts in the feed
        feed = [self.posts[post_id] for post_id in post_ids]
        # Sort the feed by timestamp
        feed.sort(key=lambda post: post.timestamp)
    
        return feed

    
    def _get_post_ids(self, uid: str, start_ts: float, end_ts: float) -> list[str]:
        return [post_id for post_id in self.user_posts[uid] if start_ts <= self.posts[post_id].timestamp <= end_ts]
