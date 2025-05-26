"""
Benchmark Results (averages over 5 iterations):
---------------------------------------------------
PyTorch (torch.save/torch.load):
  Write Time (s): 13.9192, Write Throughput (MB/s): 73.57
  Read Time  (s): 9.5387, Read Throughput  (MB/s): 107.35

NumPy (np.save/np.load):
  Write Time (s): 1.3888, Write Throughput (MB/s): 737.34
  Read Time  (s): 0.5625, Read Throughput  (MB/s): 1820.45

Memmap (np.memmap):
  Write Time (s): 2.1087, Write Throughput (MB/s): 485.60
  Read Time  (s): 0.1354, Read Throughput  (MB/s): 7562.73

HDF5 (h5py):
  Write Time (s): 1.6039, Write Throughput (MB/s): 638.43
  Read Time  (s): 0.5448, Read Throughput  (MB/s): 1879.52

Direct Binary I/O:
  Write Time (s): 2.2782, Write Throughput (MB/s): 449.47
  Read Time  (s): 0.5722, Read Throughput  (MB/s): 1789.46

"""
import os
import time

import h5py
import numpy as np
import torch

# --------------------
# Configuration
# --------------------

TENSOR_SHAPE = (1024, 1024, 256)  # shape of the test tensor
DTYPE = torch.float32  # data type for the tensor
NUM_ITERATIONS = 5  # how many times to repeat each test

# We'll store temporary files in the current directory.
# Adjust the filenames or paths as needed.
PT_FILE = "test_tensor.pt"
NPY_FILE = "test_tensor.npy"
MEMMAP_FILE = "test_tensor_memmap.dat"
H5_FILE = "test_tensor.h5"
BIN_FILE = "test_tensor.bin"

# --------------------
# Utility Functions
# --------------------


def generate_test_tensor():
    """
    Generate a random tensor of the specified shape and data type.
    """
    return torch.randn(*TENSOR_SHAPE, dtype=DTYPE)


def measure_size_in_bytes(shape, dtype):
    """
    Returns the size in bytes for a given shape and dtype.
    """
    # For PyTorch dtype, convert to NumPy dtype
    np_dtype = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }.get(dtype, np.float32)
    return np.prod(shape) * np.dtype(np_dtype).itemsize


def mbps(num_bytes, duration):
    """
    Convert `num_bytes` over `duration` seconds into MB/s.
    """
    return (num_bytes / (1024 * 1024)) / duration


# --------------------
# Benchmark Functions
# --------------------


def benchmark_torch_save_load(tensor, num_bytes):
    # --- Write ---
    start = time.time()
    torch.save(tensor, PT_FILE)
    write_time = time.time() - start

    # --- Read ---
    start = time.time()
    loaded = torch.load(PT_FILE)
    read_time = time.time() - start

    return write_time, read_time


def benchmark_numpy_save_load(array, num_bytes):
    # --- Write ---
    start = time.time()
    np.save(NPY_FILE, array)
    write_time = time.time() - start

    # --- Read ---
    start = time.time()
    loaded = np.load(NPY_FILE)
    read_time = time.time() - start

    return write_time, read_time


import subprocess


def drop_caches_linux():
    # This is a Linux-specific approach; requires root privileges
    subprocess.run(["sync"])
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])


def benchmark_memmap(array, num_bytes):
    # -- Write --
    start = time.time()
    fp = np.memmap(MEMMAP_FILE, dtype=array.dtype, mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    write_time = time.time() - start

    # Force OS to flush and drop caches (on Linux)
    drop_caches_linux()

    # -- Read --
    start = time.time()
    fp_read = np.memmap(MEMMAP_FILE, dtype=array.dtype, mode="r", shape=array.shape)
    data = fp_read[:]  # Force a copy into memory
    data_sum = data.sum()  # Force all pages to actually be accessed
    read_time = time.time() - start

    return write_time, read_time


def benchmark_hdf5_write_read(array, num_bytes):
    # --- Write ---
    start = time.time()
    with h5py.File(H5_FILE, "w") as f:
        f.create_dataset("tensor", data=array)
    write_time = time.time() - start

    # --- Read ---
    start = time.time()
    with h5py.File(H5_FILE, "r") as f:
        data = f["tensor"][:]
    read_time = time.time() - start

    return write_time, read_time


def benchmark_direct_binary(array, num_bytes):
    # --- Write ---
    start = time.time()
    with open(BIN_FILE, "wb") as f:
        f.write(array.tobytes())
    write_time = time.time() - start

    # --- Read ---
    start = time.time()
    with open(BIN_FILE, "rb") as f:
        raw = f.read()
        data = np.frombuffer(raw, dtype=array.dtype)
        data = data.reshape(array.shape)
    read_time = time.time() - start

    return write_time, read_time


# --------------------
# Main Benchmark Logic
# --------------------


def main():
    num_bytes = measure_size_in_bytes(TENSOR_SHAPE, DTYPE)
    print(f"Tensor shape: {TENSOR_SHAPE}, dtype: {DTYPE}, size: {num_bytes/1e6:.2f} MB")
    print(f"Running each test for {NUM_ITERATIONS} iterations...\n")

    benchmarks = {
        "PyTorch (torch.save/torch.load)": benchmark_torch_save_load,
        "NumPy (np.save/np.load)": benchmark_numpy_save_load,
        "Memmap (np.memmap)": benchmark_memmap,
        "HDF5 (h5py)": benchmark_hdf5_write_read,
        "Direct Binary I/O": benchmark_direct_binary,
    }

    results = {}

    for name, func in benchmarks.items():
        total_write_time = 0.0
        total_read_time = 0.0
        for _ in range(NUM_ITERATIONS):
            # Generate a fresh tensor/array for each iteration
            tensor = generate_test_tensor()
            # Convert to NumPy array if needed
            array = tensor.numpy()

            write_time, read_time = func(array, num_bytes)
            total_write_time += write_time
            total_read_time += read_time

        avg_write_time = total_write_time / NUM_ITERATIONS
        avg_read_time = total_read_time / NUM_ITERATIONS

        write_mbps = mbps(num_bytes, avg_write_time)
        read_mbps = mbps(num_bytes, avg_read_time)

        results[name] = (avg_write_time, write_mbps, avg_read_time, read_mbps)

    # Cleanup: remove files if desired
    for f in [PT_FILE, NPY_FILE, MEMMAP_FILE, H5_FILE, BIN_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print("Benchmark Results (averages over {} iterations):".format(NUM_ITERATIONS))
    print("---------------------------------------------------")
    for name, (wtime, wmbps, rtime, rmbps) in results.items():
        print(f"{name}:")
        print(f"  Write Time (s): {wtime:.4f}, Write Throughput (MB/s): {wmbps:.2f}")
        print(f"  Read Time  (s): {rtime:.4f}, Read Throughput  (MB/s): {rmbps:.2f}")
        print()


if __name__ == "__main__":
    main()
