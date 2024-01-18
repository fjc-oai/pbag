"""
Use concurrent.futures.Future to implement a simple Future class that 
provides similar semantics like folly::Future with chain of futures.

TODO: in the continuation API, distinguish empty input vs None input.
"""
import concurrent.futures
import time

class Future:

    ex_ = None

    def __init__(self, val=None, set_value=True):
        assert not isinstance(val, concurrent.futures.Future), "Not implemented"
        assert not isinstance(val, Future), "Not implemented"
        self._fut = concurrent.futures.Future()
        if set_value:
            self._fut.set_result(val)

    def then(self, fn):
        assert self._fut
        new_fut = Future(set_value=False)
        def fn_(in_fut):
            fut_ = self.ex.submit(fn, in_fut.result())
            fut_.add_done_callback(lambda f: new_fut.set_result(f))

        self._fut.add_done_callback(fn_)    
        return new_fut
    
    def result(self):
        return self._fut.result()
    

    def set_result(self, val):
        if isinstance(val, concurrent.futures.Future):
            val = val.result()
        self._fut.set_result(val)


    @property
    def ex(self):
        if Future.ex_ is None:
            Future.ex_ = concurrent.futures.ThreadPoolExecutor()
        return Future.ex_
    
def read_data(file_name):
    time.sleep(1)
    return "aaabbbccc"

def process_data(data):
    data = data + data
    return data

def analyze_data(data):
    return len(data)


def chain():
    fut = Future('data.txt')    
    fut = fut.then(read_data)
    fut = fut.then(process_data)
    fut = fut.then(analyze_data)
    res = fut.result()
    assert res == 18, f"res: {res}"

def chain_syntax_suger():
    fut = Future('data.txt')    
    fut = fut.then(read_data).then(process_data).then(analyze_data)
    res = fut.result()
    assert res == 18, f"res: {res}"

def main():
    chain()
    chain_syntax_suger()

if __name__ == "__main__":
    main()


