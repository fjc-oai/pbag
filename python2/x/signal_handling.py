import ctypes
import signal


def signal_handler(signum, frame):
    print("Signal handler called!")

signal.signal(signal.SIGINT, signal_handler)

print("Press Ctrl+C. The signal will not be processed until the C function returns.")

# Simulate an infinite native call using ctypes
libc = ctypes.CDLL(None)
libc.sleep(10**10)  # Sleep indefinitely in native code

print("This will only print after the sleep ends.")