import torch
import time


def cuda_sleep(x):
    N_CYCLES = 1_000_000_000
    torch.cuda._sleep(x * N_CYCLES)


def test_event():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    st = time.time()
    for _ in range(5):
        with torch.cuda.stream(s1):
            cuda_sleep(1)
        with torch.cuda.stream(s2):
            cuda_sleep(1)
    torch.cuda.synchronize()
    ed = time.time()
    print(f"Without synchronization duration: {ed - st}")

    st = time.time()
    event1 = torch.cuda.Event()
    event2 = torch.cuda.Event()
    for _ in range(5):
        with torch.cuda.stream(s1):
            event2.synchronize()
            cuda_sleep(1)
            event1.record()
        with torch.cuda.stream(s2):
            event1.synchronize()
            cuda_sleep(1)
            event2.record()
    torch.cuda.synchronize()
    ed = time.time()
    print(f"With synchronization duration: {ed - st}")


def main():
    assert torch.cuda.is_available()
    test_event()


if __name__ == "__main__":
    main()
