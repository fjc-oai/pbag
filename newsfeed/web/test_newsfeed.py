import time

from utils import create_users, validate_users

from newsfeed import NewsFeed


def validate_users(users: dict[str, set[str]]) -> None:
    all_users = set(users.keys())
    for user, friends in users.items():
        assert user in all_users
        assert all(friend in all_users for friend in friends)
        assert user not in friends
        for friend in friends:
            assert user in users[friend]


def test_basics():
    users = create_users()
    validate_users(users)

    nf = NewsFeed(users)

    assert nf.post("steve", "hello_from_steve") == True
    assert nf.post("angela", "hello_from_angela") == True
    assert nf.post("tianhao", "hello_from_tianhao") == True

    ed_ts = time.time()
    DUR_SEC = 100
    st_ts = ed_ts - DUR_SEC

    feed_res = nf.feed("steve", st_ts, ed_ts)
    EXPECTED = ["hello_from_steve", "hello_from_angela", "hello_from_tianhao"]
    assert feed_res == EXPECTED

    feed_res = nf.feed("angela", st_ts, ed_ts)
    EXPECTED = ["hello_from_steve", "hello_from_angela"]
    assert feed_res == EXPECTED

    feed_res = nf.feed("tianhao", st_ts, ed_ts)
    EXPECTED = ["hello_from_steve", "hello_from_tianhao"]
    assert feed_res == EXPECTED
