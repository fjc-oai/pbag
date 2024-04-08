import time

def async_execution(fn):
    time.sleep(5)
    print("Now executing the function")
    fn()
    print("Function executed")

def fn_as_param():
    print("Hello from fn_as_param")
    return 7

def my_fn(x):
    print(f"Hello from my_fn with {x=}")

def main():
    # async_execution(lambda: my_fn(fn_as_param()))
    fn = lambda x=fn_as_param(): my_fn(x)
    async_execution(fn)

if __name__ == "__main__":
    main()