# Python thread context switch overhead

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