import random
import time

from locust import HttpUser, between, task


class NewsFeedUser(HttpUser):
    # Wait time between tasks (1s to 3s). Adjust as desired.
    wait_time = between(1, 3)

    @task(1)
    def create_post(self):
        """
        Task 1: Send a POST request to the post service running on port 7008.
        """
        user = f"user_{random.randint(1, 10)}"
        content = f"Hello from {user} at {time.time()}"
        
        # Use a full URL because it's on a separate port
        url = f"http://localhost:7008/post?user={user}&content={content}"

        with self.client.post(url, name="POST /post", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(
                    f"POST failed (status code {response.status_code}): {response.text}"
                )

    @task(2)
    def fetch_feed(self):
        """
        Task 2: Send a GET request to the feed service running on port 7009.
        Weighted '2' means it will be called about twice as often as 'create_post'.
        """
        user = f"user_{random.randint(1, 10)}"
        start_ts = 0
        end_ts = time.time()  # current time
        url = f"http://localhost:7009/feed?user={user}&start_ts={start_ts}&end_ts={end_ts}"

        with self.client.get(url, name="GET /feed", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(
                    f"GET feed failed (status code {response.status_code}): {response.text}"
                )