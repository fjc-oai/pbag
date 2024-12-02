def test_functools():
    from functools import lru_cache

    class Client:
        def __init__(self):
            print("Client.__init__")
            self.x = 123

    @lru_cache()
    def get_client():
        return Client()

    c1 = get_client()
    c2 = get_client()
    assert c1 is c2

    @lru_cache(maxsize=2)
    def get_client2():
        return Client()

    clients = [get_client2() for _ in range(10)]
    for client in clients:
        assert client is clients[0]


def test_functools_1():
    from functools import lru_cache

    class Client:
        def __init__(self, x):
            self.x = x

    @lru_cache(maxsize=2)
    def get_client(x):
        return Client(x)

    client1s = [get_client(1) for _ in range(10)]
    assert all(client1s[0] is client1 for client1 in client1s)
    client2s = [get_client(2) for _ in range(10)]
    assert all(client2s[0] is client2 for client2 in client2s)
    client3s = [get_client(3) for _ in range(10)]
    assert all(client3s[0] is client3 for client3 in client3s)
    client1 = get_client(1)
    assert not client1 is client1s[0]


def main():
    test_functools_1()


if __name__ == "__main__":
    main()
