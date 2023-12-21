from lib import f 

import subprocess
import traceback

original_subprocess_call = subprocess.call

def patched_call(*args, **kwargs):
    if 'tail' in args[0]:
        print('tail is called')
        traceback.print_stack()
    return original_subprocess_call(*args, **kwargs)

def main():
    subprocess.call = patched_call
    f()

if __name__ == '__main__':
    main()