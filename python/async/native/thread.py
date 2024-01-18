"""
Simple usage of native Python threads.
Threads share the same memory space, so they can easily 
communicate with each other, as long as race condition has 
been taken care of.
"""
import threading

def worker(idx, res):
    res[idx] = idx * idx

def test():
    N = 5
    res = [None] * N
    threads = [threading.Thread(target=worker, args=(x, res)) for x in range(N)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert res == [x * x for x in range(N)]

def main():
    test()

if __name__ == "__main__":
    main()  