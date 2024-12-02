"""
Simple usage of native Python processes.
"""
import multiprocessing


def compute_square(idx, result):
    result[idx] = idx * idx


def test():
    N = 5
    # Shared memory for results
    result = multiprocessing.Array("i", N)
    processes = [multiprocessing.Process(target=compute_square, args=(i, result)) for i in range(N)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert list(result) == [x * x for x in range(N)]


def main():
    test()


if __name__ == "__main__":
    main()
