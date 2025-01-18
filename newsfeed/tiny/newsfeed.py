import time


class Newsfeed:
    def __init__(self, users: dict[str, set[str]]) -> None:
        self.users = users
        self.user_posts: dict[str, list[str]] = {}
        self.posts: dict[str, tuple[str, str, float]] = {}


    def post(self, uid: str, message: str) -> bool:
        timestamp = float(time.time())
        post_id = f"{uid}-{timestamp}"
        self.posts[post_id] = (uid, message, timestamp)
        if uid not in self.user_posts:
            self.user_posts[uid] = [post_id]
        else:
            self.user_posts[uid].append(post_id)
        return True


    def feed(self, uid: str, start_ts: float, end_ts: float) -> list[str]:
        post_ids = []
    
        # Get the list of users that the user follows
        following = self.users.get(uid, set())
    
        # Get the list of posts that the user has made between the start and end timestamps
        post_ids += self._get_posts(uid, start_ts, end_ts)
 
        # Get the list of posts that the user follows
        for user in following:
            post_ids += self._get_posts(user, start_ts, end_ts)

        # Sort the posts by timestamp
        post_ids.sort(key=lambda post_id: self.posts[post_id][2], reverse=True)
        # Return the list of posts in the feed
        feed = [self.posts[post_id][1] for post_id in post_ids]
    
        return feed

    
    def _get_posts(self, uid: str, start_ts: float, end_ts: float) -> list[str]:
        user_posts = self.user_posts.get(uid, [])
        return [post_id for post_id in user_posts if start_ts <= self.posts[post_id][2] <= end_ts]
