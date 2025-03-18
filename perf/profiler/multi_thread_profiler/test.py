import contextlib
import json
import threading
import cprofiler
import time
from tqdm import tqdm
import torch
@contextlib.contextmanager
def profiler():
    yield
    return
    cprofiler.set_profiler()
    yield
    cprofiler.unset_profiler()

def task(duration):
    time.sleep(duration * 0.1)


def attn():
    task(1)


def mlp():
    task(2)


def fwd_layer():
    attn()
    mlp()


def bwd_layer():
    mlp()
    attn()


n_layers = 16
fwd_done = [threading.Event() for _ in range(n_layers)]
bwd_done = [threading.Event() for _ in range(n_layers)]


def fwd():
    with profiler():
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            all_threads=True,
        )
        prof.start()
        for i in range(16):
            if i > 0:
                bwd_done[i - 1].wait()
            fwd_layer()
            fwd_done[i].set()
        prof.stop()
        prof.export_chrome_trace("/root/torch_trace.json")
        print(f"Trace file generated: /root/torch_trace.json")


def bwd():
    with profiler():
        for i in range(16):
            fwd_done[i].wait()
            bwd_layer()
            bwd_done[i].set()


def generate_trace_file(profiler_data, output_file):
    # Extract the file and function name arrays.
    pos2_file = profiler_data["pos_2_file_name"]
    pos2_func = profiler_data["pos_2_func_name"]
    thread_stats = profiler_data["thread_stats"]

    trace_events = []

    # Iterate over each thread in the stats.
    for tid, records in thread_stats.items():
        for rec in records:
            file_index, func_index, line_no, start, end = rec

            event = {
                "name": pos2_func[func_index],  # Use the function name.
                "cat": "function",
                "ph": "X",  # 'X' stands for complete events.
                "ts": start,  # Start timestamp (in microseconds or any time unit you use).
                "dur": end - start,  # Duration of the function call.
                "pid": 0,  # Process id (set to 0 if there's only one process).
                "tid": tid,  # The thread id.
                "args": {"file": pos2_file[file_index], "line": line_no},
            }
            trace_events.append(event)

    trace_data = {"traceEvents": trace_events}

    # Write the JSON trace file.
    with open(output_file, "w") as f:
        json.dump(trace_data, f, indent=2)
    print(f"Trace file generated: {output_file}")


def generate_flamegraph_file(profiler_data, output_file):
    pos2_func = profiler_data["pos_2_func_name"]
    thread_stats = profiler_data["thread_stats"]

    # Open the output file for writing
    with open(output_file, "w") as f:
        # Process each thread separately
        for tid, events in thread_stats.items():
            # Sort events by start time
            events.sort(key=lambda rec: rec[3])
            stack = []
            # Process each event in order; assume events are properly nested.
            for rec in events:
                file_index, func_index, lineno, st, ed = rec
                # Pop from the stack until the new event is nested in the current top.
                while stack and st >= stack[-1][4]:
                    stack.pop()
                # Push the current event onto the stack.
                stack.append(rec)
                # The call chain is the sequence of function names in the current stack.
                call_chain = ";".join(pos2_func[frame[1]] for frame in stack)
                # Use the duration of the current event as the metric.
                duration = ed - st
                # Write a line in the folded stack format.
                f.write(f"{call_chain} {duration}\n")
    print(f"Flamegraph file generated: {output_file}")


def test():
    with profiler():
        threads = [
            threading.Thread(target=fwd),
            threading.Thread(target=bwd),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    res = cprofiler.dump_stats()
    generate_trace_file(res, "trace.json")
    generate_flamegraph_file(res, "flamegraph.folded")


# test()

def test_2_with_torch_profiler():
    # prof = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     all_threads=True,
    # )
    threads = [
        threading.Thread(target=fwd),
        threading.Thread(target=bwd),
    ]
    # prof.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # prof.stop()
    # prof.export_chrome_trace("/root/torch_trace.json")
    # print(f"Trace file generated: /root/torch_trace.json")

test_2_with_torch_profiler()