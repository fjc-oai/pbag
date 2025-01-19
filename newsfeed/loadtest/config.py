from dataclasses import dataclass
from typing import literal

SERVICE_HOST = "0.0.0.0"
WEB_SERVICE_PORT = 7007
POST_SERVICE_PORT = 7008
FEED_SERVICE_PORT = 7009


@dataclass
class UserConfig:
    mode: str  # ["real", "loadtest"]
    n_users: int = 0
    avg_n_friends: int = 0

    def __post_init__(self):
        assert self.mode in ["real", "loadtest"], f"Invalid mode: {self.mode}"
        assert self.n_users >= 0, f"Invalid number of users: {self.n_users}"
        assert self.avg_n_friends >= 0, f"Invalid average number of friends: {self.avg_n_friends}"
        assert (
            self.avg_n_friends < self.n_users
        ), f"Average number of friends must be less than number of users"


global_user_config = None

real_user_config = UserConfig(mode="real")
small_loadtest_user_config = UserConfig(mode="loadtest", n_users=100, avg_n_friends=10)
mid_loadtest_user_config = UserConfig(mode="loadtest", n_users=1000, avg_n_friends=100)
large_loadtest_user_config = UserConfig(mode="loadtest", n_users=1_000_000, avg_n_friends=1000)


def get_user_config() -> UserConfig:
    global global_user_config
    if global_user_config is None:
        global_user_config = small_loadtest_user_config
    return global_user_config
