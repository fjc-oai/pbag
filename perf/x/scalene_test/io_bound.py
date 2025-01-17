import time


def write_large_file(filename, lines, thread_id):
    """Write a large file (simulating I/O-bound work)."""
    print(f"[Thread {thread_id}] Writing {lines} lines to {filename}...")
    with open(filename, "w") as f:
        for i in range(lines):
            f.write(f"Thread {thread_id} - Line {i}: Test line.\n")

def read_large_file(filename, thread_id):
    """Read a large file line by line."""
    print(f"[Thread {thread_id}] Reading {filename}...")
    with open(filename, "r") as f:
        for line in f:
            _ = line.strip()  # Simulate processing