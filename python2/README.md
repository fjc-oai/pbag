# Thread

##  Context switch 

1. What is a python thread? Is it a pure application level object?
    - No. It uses OS thread under the hood
    - A normal OS thread can start execution as long as its state becomes to ready-to-run
    - A python thread essentially executes python byte-code. In order to exectute, it first requires underlying OS thread is ready-to-run, and then requires to acquare GIL.

2. How does python thread scheduling & context switch work? 
    - Voluntary (yield) and non-voluntary context switch (preemption)
    - Yield: sleep(), io.wait(), lock.acquire(), etc
    - Preemption: interpreter forces a context switch on a thread after a certain number of byte-code executed, which is controlled by `sys.setswitchinterval()`.

3. Why excessive theads can slow down the program?
    - More threads compete for GIL
    - More threads to schedule on OS

4. How to benchmark context switch overhead?
    -  `python context_switch_overhead.py`
    - Complex arithmatic operations (hard to opt away), memory access, etc.
    - Otherwise won't be easy to repro with loop over simple operations (e.g. float mul only)

5. Why profiler doesn't help in this case?
    - Profiler, e.g. py-spy doesn't show context switch overhead
    - It has no visibility into system-level scheduling, and only samples python-level threads, but not OS level.


## Thread creation overhead

- Benchmark: `python thread_creation_overhead.py`

- Always create a new thread on-the-fly vs using a global thread pool executor

- Intuitively, thread creation itself is expensive. So it'll be slower (plus thread context switch overhead).

- From benchmark, the difference is not that significant though, ~10%.



## Thread in Asyncio 

- Demo: `python concurrent_future_in_asyncio.py`

- Works in a almost the same way as io/select implementation

- `run_in_executor()` runs the enclosed task in the executor on  one hand, it wraps the returned `concurrent.Future` with `asyncio.Future` which monitored by eventloop on the other hand. 

- Once the underlyting `concurrent.Future` is completed, the corresponding `asyncio.Future` will be marked done as well, thus be ready to be picked up by eventloop to schedule remaining code in its upstream task.

- If underlying function runs asynchronously and returns a concurrent.futures.Future, extra attention is required!
    1. asyncio.wrap_future() to wrap the returned concurrent.futures.Future and await it
    2. try-except the block, and propagate the execution properly
    3. Demo: `python wrap_future.py`

# Context Var

1. Basics
   - Manage context-local state, used mainly in async programming (e.g. asyncio) 
   - Each context (e.g. coro/async task) has its own value for a variable
   - Similar to request context commonly used in C++ backend services
2. Context var vs thread-local
   - Context var: mainly used in asyncio. All the tasks run on the same threads
   - Thread-local: mainly used in multi-threading. Each task runs on its own thread


# Itertools

1. `it1 = itertools.count(start=1)`

2. `it2 = more_itertools.batched(it1, 5)`