"""
When using pdb, breakpoint() will pause only the current thread while others
threads are still running. Similar to debugpy.
"""
import concurrent.futures
import time


def bp():
    import pdb
    pdb.set_trace()
    # import debugpy
    # debugpy.listen(5678)
    # debugpy.wait_for_client()

def worker(idx):
    st = time.time()
    while True:
        print(f"Worker {idx} is working")
        time.sleep(1)
        if idx == 3 and time.time() - st > 5:
            bp()



def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futs = [executor.submit(worker, i) for i in range(5)]
        for fut in futs:
            fut.result()

if __name__ == "__main__":
    main()