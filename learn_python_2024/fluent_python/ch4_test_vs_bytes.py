import pytest


def test_encode_decode():
    s = "café"
    assert type(s) == str
    assert len(s) == 4

    b = s.encode("utf-8")
    assert type(b) == bytes
    assert len(b) == 5

    s2 = b.decode("utf-8")
    assert type(s2) == str


def test_bytes_display():
    b = bytes([0x41, 0x42, 0x43, 0x44, 0x88, 0x0A, 0x0B, 0xAA, 0xAB])
    print(b)


def test_read_write_file(tmpdir):
    file = tmpdir.join("test")
    with open(file, "w", encoding="utf-8") as f:
        f.write("café")

    with open(file, "wb") as f:
        f.write("café".encode("utf-8"))
        with pytest.raises(TypeError):
            f.write("café")

    with open(file, "rb") as f:
        content = f.read()
        assert type(content) == bytes
        assert content == b"caf\xc3\xa9"

    with open(file, "r") as f:
        content = f.read()
        assert type(content) == str
        assert content == "café"
