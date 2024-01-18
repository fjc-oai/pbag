"""
concurrent.futures.wait() provides similar semantics like future.collect() and future.collect_all() in folly.
It seems the returns future can be out of order though.
"""
import concurrent.futures


def task(x):
    return x * x


def test():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futs = [executor.submit(task, x) for x in range(3)]
    done, not_done = concurrent.futures.wait(futs, return_when=concurrent.futures.ALL_COMPLETED)
    print(f"done: {len(done)}")
    print(f"not_done: {len(not_done)}")
    for fut in done:
        print(fut.result())


def main():
    test()


if __name__ == "__main__":
    main()
