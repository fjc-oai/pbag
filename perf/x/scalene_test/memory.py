import threading

import numpy as np


def generate_large_array(size, thread_id):
    """Generate a large NumPy array (Memory-heavy)."""
    print(f"[Thread {thread_id}] Generating {size}x{size} array...")
    arr = np.random.rand(size, size)
    return arr

def process_array(arr, thread_id):
    """Process array (e.g., compute column means)."""
    print(f"[Thread {thread_id}] Processing array...")
    return arr.mean(axis=0)