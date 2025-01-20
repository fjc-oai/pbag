from dataclasses import dataclass

SERVICE_HOST = "0.0.0.0"
ANOTHER_HOST = "192.168.1.194"
WEB_SERVICE_PORT = 7007
FEED_SERVICE_PORT = 7008
POST_SERVICE_PORT = 7009


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
            self.avg_n_friends <= self.n_users
        ), f"Average number of friends must be less than number of users {self.avg_n_friends} < {self.n_users}"


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


LOADTEST_CLIENT_TIMEOUT_S = 2
WEB_SERVICE_N_WORKERS = 4
POST_SERVICE_N_WORKERS = 1
POST_BURN_CPU_MS = 20
FEED_BURN_CPU_MS = 100


@dataclass(frozen=True)
class Server:
    host: str
    port: int

post_service_servers = [
    Server(SERVICE_HOST, POST_SERVICE_PORT),
    Server(SERVICE_HOST, POST_SERVICE_PORT + 1),
    Server(SERVICE_HOST, POST_SERVICE_PORT + 2),
    Server(SERVICE_HOST, POST_SERVICE_PORT + 3),
    # Server(ANOTHER_HOST, POST_SERVICE_PORT + 1),
    # Server(ANOTHER_HOST, POST_SERVICE_PORT + 2),
    # Server(ANOTHER_HOST, POST_SERVICE_PORT + 3),
]