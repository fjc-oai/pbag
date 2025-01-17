import math
import threading


def is_prime(n):
    """Check if a number is prime (CPU-bound work)."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def count_primes(n, thread_id):
    """Counts prime numbers up to n."""
    print(f"[Thread {thread_id}] Counting primes up to {n}...")
    primes = [x for x in range(n) if is_prime(x)]
    print(f"[Thread {thread_id}] Found {len(primes)} prime numbers.")
    return len(primes)