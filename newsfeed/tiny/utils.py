def create_users() -> dict[str, set[str]]:
    users: dict[str, set[str]] = {
        "steve": {"angela", "zichen", "mengdi", "tianhao"},
        "angela": {"steve", "zichen", "mengdi", "zhulin"},
        "zichen": {"steve", "angela", "mengdi"},
        "mengdi": {"steve", "angela", "zichen"},
        "tianhao": {"steve"},
        "zhulin": {"angela"},
    }
    return users

def validate_users(users: dict[str, set[str]]) -> None:
    all_users = set(users.keys())
    for user, friends in users.items():
        assert user in all_users
        assert all(friend in all_users for friend in friends)
        assert user not in friends
        for friend in friends:
            assert user in users[friend]