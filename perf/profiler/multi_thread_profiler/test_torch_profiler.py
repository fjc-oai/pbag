import time
import json
import torch

import cprofiler

cprofiler.set_profiler()

prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    all_threads=True,
)

prof.start()

def task():
    time.sleep(3)

task()

prof.stop()

cprofiler.unset_profiler()
res = cprofiler.dump_stats()

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


generate_trace_file(res, "/tmp/tp_trace.json")
