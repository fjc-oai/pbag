"""
/dev/shm: Why?
1. /dev/shm VS disk file: 
    - RAM backed. Faster access, though volatile (cleared on reboot)
2. /dev/shm VS in-memory data structure: 
    - Persistent after program restarts. 
    - Shared acorss processes. Easy IPC. Use fcntl for synchronization


mmap: What?
1. Map a file into memory address (newly created virtual memory region)
2. Zero copy to create nparray and torch.tensor
3. Use close() or flush() to write back to file, if MAP_SHARED. COW if MAP_PRIVATE
"""

import concurrent.futures
import fcntl
import mmap

import numpy as np
import torch


def test_access_shm_from_multiprocess():
    def write_shm(rank: int):
        with open("/dev/shm/hellos", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f"hello from {rank}\n")
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"written by {rank}")

    futs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for rank in range(8):
            fut = executor.submit(write_shm, rank)
            futs.append(fut)

    for fut in futs:
        fut.result()


def test_mmap_access_shm_from_multiprocess():
    def write_shm_mmap(rank: int):
        with open("/dev/shm/hellos2", "r+b") as f:
            with mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED) as mm:
                msg = f"hello from {rank}\n".encode()
                off = 0 + rank * len(msg)
                mm.seek(off, 0)
                mm.write(msg)
                mm.close()
        print(f"written by {rank}")

    with open("/dev/shm/hellos2", "wb") as f:
        f.write(b" " * 1024)
    futs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for rank in range(8):
            fut = executor.submit(write_shm_mmap, rank)
            futs.append(fut)

    for fut in futs:
        fut.result()


def test_mmap_as_tensor():
    with open("/dev/shm/tensors", "wb") as f:
        data = np.arange(32, dtype=np.float32)
        f.write(data.tobytes())

    with open("/dev/shm/tensors", "r+b") as f:
        with mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED) as mm:
            array = np.ndarray(shape=(32,), dtype=np.float32, buffer=mm)
            tesnor = torch.tensor(array)
            print(tesnor)
