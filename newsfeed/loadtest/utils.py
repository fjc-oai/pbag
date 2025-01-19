import random

from config import get_user_config


def create_users() -> dict[str, set[str]]:
    user_config = get_user_config()
    if user_config.mode == "real":
        return create_real_users()
    elif user_config.mode == "loadtest":
        return create_fake_users(user_config.n_users, user_config.avg_n_friends)
    else:
        raise ValueError(f"Invalid mode: {user_config.mode}")


def create_real_users() -> dict[str, set[str]]:
    users: dict[str, set[str]] = {
        "steve": {"angela", "zichen", "mengdi", "tianhao"},
        "angela": {"steve", "zichen", "mengdi", "zhulin"},
        "zichen": {"steve", "angela", "mengdi"},
        "mengdi": {"steve", "angela", "zichen"},
        "tianhao": {"steve"},
        "zhulin": {"angela"},
    }
    return users


def create_fake_users(N: int, K: int) -> dict[str, set[str]]:
    """
    Create an undirected user friendship graph.

    :param N: Number of users
    :param K: Desired average number of friends (approx)
    :return: A dict where keys = "user_1", "user_2", ... "user_N"
             and values = set of friend user_ids (e.g. {"user_3", "user_5"}).
    """
    # Initialize each user's adjacency list as an empty set
    graph = {}
    for i in range(N):
        user_id = f"user_{i+1}"
        graph[user_id] = set()

    # Total edges we want ~ N*K/2 (to achieve average degree = K in an undirected graph)
    M = (N * K) // 2  # integer approximation

    # Generate all possible user pairs (i < j to avoid duplicates in undirected graph)
    possible_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            possible_pairs.append((i, j))

    # Shuffle and pick up to M pairs
    random.shuffle(possible_pairs)
    chosen_pairs = possible_pairs[:M]

    # Add edges to the graph
    for i, j in chosen_pairs:
        u1 = f"user_{i+1}"
        u2 = f"user_{j+1}"
        graph[u1].add(u2)
        graph[u2].add(u1)

    return graph


def validate_users(users: dict[str, set[str]]) -> None:
    all_users = set(users.keys())
    for user, friends in users.items():
        assert user in all_users
        assert all(friend in all_users for friend in friends)
        assert user not in friends
        for friend in friends:
            assert user in users[friend]
