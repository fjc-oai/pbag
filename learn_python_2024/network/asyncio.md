# Tiny-asyncio
Implement asyncio from scatch, in 400 lines of code. Build your own `await`, `sleep`, `gather`, etc. No more magic! Don't miss Q&A section. The coolest learnings. 

This doc is written with the help of ChatGPT. Similarly for the code.

## Usage Recap 
- https://docs.python.org/3/library/asyncio.html
```
import asyncio

async def main():
    data = await fetch_data()
    res = await process_data(data)
    await save_result(res)

if __name__ == "__main__":
    asyncio.run(main())

```

## Keyword Under the Hood
### async
`async` defines a function as coroutine. Think of as a generator. It can be paused and resumed, allowing for cooperative multitasking. When invoked (e.g. `async def foo(); coro=foo()`), `foo()` doesn't execute but returns a `coroutine` object (or a `async generator`).

### await
`await` pauses the execution of current coroutine, until the awaited coroutine completes. This allows the event loop to run other tasks while waiting, thus achieving concurrency.

Technically it's not equavallent but you can think of `await` as `yield from` in mental model. So, what is `yield from`? 
```
def foo():
    yield from my_generator()
```
is (largely) equivalent to 
```
def foo():
    for value in my_generator():
        yield value
```
Or (forget about `send` for now)
```
def foo():
    it = iter(my_generator())
    while True:
        try:
            value = next(it)
            yield value
        except StopIteration:
            break
```
If you want to know what exactly does python interpretor handles `yield from`, read Fluent Python chatper 16. It has a detailed manual implementation.

Now back to `await`. `await` can only be used on `awaitable`s (anything defines `__await__` can be an `awaitable`). If `await` on a coroutine object, it's equavallent to invokes its underlying generator. e.g. 
```
async def foo():
    yield 1
    yield 2
    yield 3

async def main():
    await foo()
```
It's like `yield from foo()`, though note that now the caller is `eventloop`, and it's `eventloop` received those values (e.g. 1,2,3). Similarly, it can also `await` on a customized class which implements `__await__`, e.g.
```
class Future:
    ...
    def __await__(self):
        if not self.done():
            self._asyncio_future_blocking = True
            yield self
        assert self.done(), "Future should be done"
        return self.result()
```
This is the real implementation of `Future`! You may find `self._asyncio_future_blocking = True` part confusing. It's ok for now. But remember, this is the core trick that creates the magic asynchrous-ness of asyncio!

## Major Abstractions
### Eventloop
The event loop continuously runs, monitoring for I/O events, executing scheduled tasks, and invoking callbacks. It ensures non-blocking behavior by allowing other tasks to execute while waiting for I/O operations to complete. Read `Eventloop::_run_once()` as an entry point.

### Future
An awaitable object that represents a pending result. Futures are typically used for coordination between coroutines and can be explicitly awaited to retrieve their results once the underlying operation completes. Most (if not all) of the asynchronous-ness is achieved by `Future.__await__()`!

### Task
Task inherits from `Future`.  A Task is an abstraction that wraps a coroutine, enabling it to be scheduled for execution by the event loop. `asyncio.sleep()` is pretty much all implemented by `Task::__step()`. So cool!

### Handle
A Handle is an abstraction representing a scheduled callback. Think of it as a `callable`. That's it. `TimedHandle` is a bit more interesting!

## Asyncio from Scratch!
### Plain tasks & asyncio.run()
### asyncio.sleep()
### io & select()
### periodical background task
### asyncio.gather()


## Q&As - the coolest section!
    """
    step 1: create a future
    step 2: register a call_later event, which set_result to the future
    step 3: current coro will wait for this future

    Qs:
    - if a coro can await on a future, what does future's __await__ do?
        A:  stop existing coro (by not registering it further to the event loop)
            register a new time event, happens after the delay, which set the future's result
            yield the result to coro.step() with a flag (e.g. _asyncio_future_blocking), so that coro.step() register a callback to the future to wake up/resume the coro
    - how to make a coro waiting for sth? by setting its status as pending?
       - e.g. hwo to avoid eventloop keept polling the coro to check whether it's ready
       - likely by setting the coro's status as pending, and future compleletion will somehoe changes the status
       A: it doesn't further register the event to the eventloop, until the future is done
    - future vs task:
        - future: general awaitable object
        - task: coro which can be called with send(None)?

    - eventloop.run_until_complete(task) uses task's completion to stop the eventloop. before it stops, it keeps polling the eventloop to run the ready events
        does it mean the eventloop is busy looping until the task is done? thus will take 100% CPU? assuming select timeout is 0, and there is only one sleep event scheduled.
    """

## Appendix 1 - socket programming
## Appendix 2 - non-blocking socket & selector 