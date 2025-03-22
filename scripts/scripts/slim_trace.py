"""
Torch profiler results sometimes can be tedious to read - Irrelevant threads
appearing between the main thread and cuda stream events. - Excessively deep
stack frames made cuda stream far from python functions. - Generated json files
contain invalid characters that prevent chrome://tracing/ from loading.

Usage:
    python slim_trace.py /path/to/torch_profiler.json

Options (default off for all):
    -f, --fix-invalid-chars: Automatically fix the invalid characters in the json
    file.

    -t, --trim-tids: Comma separated tids to keep, or 'auto', or '' as no-op.
    Trim irrelevant threads. Doesn't save much space but makes the trace much
    easier to read.

    -n, --trim-frame-every-n: Trim frame and keep one out of n frames. 0 as no-op.
    Last resort. Save significantly.

    --trim-python-function-args: Trim the python function args.
    Don't why those fields exist and it doesn't seem to break anything after
    trimming. Should be safe to turn on. Saved a lot of space.

    --trim-python-function-name: Trim the python function name. Show only file 
    name, lineno, func name. Save a little.


Visualization tools:
    1. chrome://tracing/
    2. https://ui.perfetto.dev/ (slower to load but much smoother
    to interact with!)


########################################
#   Data format of a trace file:
########################################

1. A dict with dict_keys(['schemaVersion', 'deviceProperties', 'with_stack',
   'traceEvents', 'traceName'])
2. 'traceEvents' is what we are interested in, a list of events
3. Each event is a dict
4. Top level categories are based on 'ph' key, e.g. 's', 'X', 'f', 'i', 'M'
    1. 's', 'f': start and finish of asynchronous flows, i.e. ac2g
    2. 'X': duration events, which are the majors ones we're interested and
       wanted to manipulate
    3. 'i': instantaneous events, e.g. tracing start and end
    4. 'M': metadata about processes/threads, e.g. used for labeling and
       ordering
5. 'X' event can be further categorized by 'cat' key, e.g. 'ac2g', 'gpu_memset',
   'cuda_runtime', 'kernel', 'Trace', 'gpu_memcpy', 'python_function',
   'cuda_driver', 'cpu_op', 'fwdbwd'
    1. 'ac2g': ac2g event
    2. python_function: cpu side python functions
    3. 'cuda_runtime': cpu side op that invoking cuda kernels
    4. 'kernel': cuda kernel
    5. 'cpu_op': cpu side annotations
    6. Others: gpu_memcpy, cpu_op, fwdbwd, gpu_memset, python_function

    ########################################
    # Example of a cuda kernel event # cuda_events = [e for e in events if
    e['ph']=='X' and e['cat']=='kernel'] # pp cuda_events[0] {'args': {'External
    id': 311878,
            'block': [512, 1, 1], 'blocks per SM': 4.0, 'context': 1,
            'correlation': 578471, 'device': 0, 'est. achieved occupancy %':
            100, 'grid': [264, 2, 1], 'queued': 0, 'registers per thread': 16,
            'shared memory': 0, 'stream': 7, 'warps per SM': 64.0},
    'cat': 'kernel', 'dur': 3.36, 'name': 'void at::native::(anonymous '
            'namespace)::CatArrayBatchedCopy<at::native::(anonymous '
            'namespace)::OpaqueType<8u>, unsigned int, 1, 64, '
            '64>(at::native::(anonymous namespace)::OpaqueType<8u>*, '
            'at::native::(anonymous '
            'namespace)::CatArrInputTensorMetadata<at::native::(anonymous '
            'namespace)::OpaqueType<8u>, unsigned int, 64, 64>, '
            'at::native::(anonymous namespace)::TensorSizeStride<unsigned int, '
            '4u>, int, unsigned int)',
    'ph': 'X', 'pid': 0, 'tid': 7, 'ts': 3777862676194.833} }

    ########################################
    # Example of a python function event # python_events = [e for e in events if
    e['ph']=='X' and e['cat']=='python_function'] # pp python_events[0] {'args':
    {'Ev Idx': 321951, 'Python id': 1, 'Python parent id': None}, 'cat':
    'python_function', 'dur': 79383.308, 'name': 'beam/invokers.py(325):
    call_rpc', 'ph': 'X', 'pid': 36637, 'tid': 49121, 'ts': 3777853151615.0} }


6. Python side dependencies:
    1. each python_function event has a 'Python id' and 'Python parent id'
    2. though the dependency (i.e. stacktrace ordering) is based on 'ts' and
       'dur' keys
7. CPU <> GPU dependencies:
    1. Each cpu op invoking cuda kernel has a correlation id, e.g. `{'ph': 'X',
       'cat': 'cuda_runtime', 'name': 'cudaLaunchKernelExC', 'pid': 206656,
       'tid': 207287, 'ts': 1733531467127860, 'dur': 15, 'args': {'External id':
       32731, 'cbid': 430, 'correlation': 32731}}`
    2. Similar to GPU kernel ops
8. Thread and process rendering order:
    1. 'M' events include the ordering, e.g. `{'name': 'thread_sort_index',
       'ph': 'M', 'ts': 1733531465924969, 'pid': 0, 'tid': 7, 'args':
       {'sort_index': 7}}`


"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Args:
    path: str
    fix_invalid_chars: bool
    trim_python_function_args: bool
    trim_python_function_name: bool
    trim_tids: str
    trim_frame_every_n: int


def process(events: list, args: Args) -> list:
    output_events = []
    phases = {e["ph"] for e in events}
    expected_phases = {"s", "X", "f", "i", "M"}
    assert (
        phases - expected_phases == set()
    ), f"Unexpected phases: {phases - expected_phases}"
    phase_to_events = defaultdict(list)
    for event in events:
        phase_to_events[event["ph"]].append(event)
    print("*" * 100)
    print("Number of events of each phase:")
    for phase, events in phase_to_events.items():
        print(f"    {phase=}: {len(events)}")

    for phase, events in phase_to_events.items():
        if phase != "X":
            output_events += events

    X_events = phase_to_events["X"]
    print(f"Total number of X events: {len(X_events)}")

    if args.trim_python_function_args:
        X_events = _trim_python_function_args(X_events)

    if args.trim_python_function_name:
        X_events = _trim_python_function_name(X_events)

    if args.trim_tids != "":
        X_events = _trim_tid_pid(X_events, args.trim_tids)

    if args.trim_frame_every_n > 0:
        X_events = _trim_frame_every_n(X_events, args.trim_frame_every_n)

    output_events += X_events
    return output_events


def _trim_python_function_args(events: list) -> list:
    print("*" * 100)
    print(f"Trimming python function args...")
    cnt = 0
    for e in events:
        if e["ph"] == "X":
            if e["cat"] == "python_function":
                e["args"] = {}
                cnt += 1
    print(f"Trimmed python function args of {cnt} events")
    return events


def _trim_python_function_name(events: list) -> list:
    print("*" * 100)
    print(f"Trimming python function name...")
    cnt = 0
    pattern = r"([^/]+\.py\(\d+\): \S+)"
    for e in events:
        if e["ph"] == "X":
            if e["cat"] == "python_function":
                name = e["name"]
                match = re.search(pattern, name)
                if match:
                    e["name"] = match.group(1)
                    cnt += 1
    print(f"Trimmed python function name of {cnt} events")
    return events


def _trim_tid_pid(events: list, tids: str) -> list:
    if tids == "":
        return events
    print("*" * 100)
    print(f"Trimming tid_pid...")
    assert all("pid" in e and "tid" in e for e in events)
    pid_tids = {(e["pid"], e["tid"]) for e in events}
    print(f"Total number of threads: {len(pid_tids)}")
    fwd_bwd_pid_tids = set()
    gpu_pid_tids = set()
    for event in events:
        cat = event.get("cat", "")
        if cat == "kernel":
            gpu_pid_tids.add((event["pid"], event["tid"]))
        elif cat == "cuda_runtime":
            name = event.get("name", "")
            if name not in ["cudaEventQuery", "cudaEventElapsedTime"]:
                fwd_bwd_pid_tids.add((event["pid"], event["tid"]))
    print(f"    fwd_bwd_pid_tids {len(fwd_bwd_pid_tids)}: {fwd_bwd_pid_tids}")
    print(f"    gpu_pid_tids {len(gpu_pid_tids)}: {gpu_pid_tids}")
    pid_tids_to_keep = set()
    if tids == "":
        pid_tids_to_keep = pid_tids
    elif tids == "auto":
        pid_tids_to_keep = gpu_pid_tids | fwd_bwd_pid_tids
    else:
        tids = [int(tid) for tid in tids.split(",")]
        pid_tids_to_keep = {(pid, tid) for pid, tid in pid_tids if tid in tids}
    print(
        f"    Keep {len(pid_tids_to_keep)} useful threads, trimming {len(pid_tids - pid_tids_to_keep)} trivial threads..."
    )
    new_events = [e for e in events if (e["pid"], e["tid"]) in pid_tids_to_keep]
    print(f"Trimmed {len(events) - len(new_events)} events")
    return new_events


def _trim_frame_every_n(events: list, n: int) -> list:
    if n == 0:
        return events
    print("*" * 100)
    print(f"Trimming frame every {n}...")
    new_events = []
    pid_tids = {(e["pid"], e["tid"]) for e in events}
    for pid_tid in pid_tids:
        pid_tid_events = [e for e in events if (e["pid"], e["tid"]) == pid_tid]
        python_pid_tid_events = [
            e for e in pid_tid_events if e["cat"] == "python_function"
        ]
        other_pid_tid_events = [
            e for e in pid_tid_events if e["cat"] != "python_function"
        ]
        new_python_pid_tid_events = per_thread_trim_frame_every_n(
            python_pid_tid_events, n
        )
        new_events += other_pid_tid_events + new_python_pid_tid_events
        print(
            f"Trim frame {pid_tid}: {len(python_pid_tid_events)} -> {len(new_python_pid_tid_events)}"
        )
    print(f"Trimmed {len(events) - len(new_events)} events")
    return new_events


def per_thread_trim_frame_every_n(events: list, n: int) -> list:
    assert all(e["cat"] == "python_function" for e in events)
    assert all("args" in e for e in events)
    assert all("Python id" in e["args"] for e in events)
    assert all("ts" in e for e in events)

    root_nodes = build_tree(events)
    nodes = root_nodes[:]
    new_events = []
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        if node.lvl % n == 0:
            node.mark = True
            new_events.append(node.event)
        else:
            node.mark = False
    return new_events


@dataclass
class Node:
    event: dict
    children: list["Node"]
    parent: "Node"
    lvl: int
    mark: bool = False


def build_tree(events):
    id_to_node = {}
    events = sorted(
        events, key=lambda e: (e["ts"], e["ts"] - e.get("dur", 0))
    )  # Parent event goes first
    root_nodes = []
    for event in events:
        node = Node(event=event, children=[], parent=None, lvl=0)
        id_to_node[event["args"]["Python id"]] = node
        parent_id = event["args"].get("Python parent id", None)
        if parent_id is None:
            root_nodes.append(node)
        else:
            if parent_id not in id_to_node:
                # Parent event is from another thread
                root_nodes.append(node)
            else:
                parent_node = id_to_node[parent_id]
                node.parent = parent_node
                node.lvl = parent_node.lvl + 1
                parent_node.children.append(node)
    return root_nodes


def _load_json(args: Args):
    path = args.path
    if path.startswith("~"):
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if args.fix_invalid_chars:
        print(f"Now loading {path} with invalid characters fixed...")
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        # Remove all non-ASCII characters
        sanitized_content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", content)
        sanitized_content = re.sub(r"[^\x00-\x7F]", "", sanitized_content)
        try:
            data = json.loads(sanitized_content)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for {path} due to {e}")
            raise e
    else:
        with open(path, "r") as f:
            data = json.load(f)
    return data


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str)
    argparser.add_argument(
        "--fix-invalid-chars",
        "-f",
        type=bool,
        default=False,
        help="Fix invalid characters in the json file",
    )
    argparser.add_argument(
        "--trim-python-function-args",
        type=bool,
        default=True,
        help="Trim python function args",
    )
    argparser.add_argument(
        "--trim-python-function-name",
        type=bool,
        default=False,
        help="Trim python function name",
    )
    argparser.add_argument(
        "--trim-tids",
        "-t",
        type=str,
        default="",
        help="Comma separated tids to keep, or 'auto', or '' as no-op",
    )
    argparser.add_argument(
        "--trim-frame-every-n",
        "-n",
        type=int,
        default=0,
        help="Trim frame and keep one out of n frames. 0 as no-op",
    )
    args = argparser.parse_args()

    args = Args(
        path=args.path,
        fix_invalid_chars=args.fix_invalid_chars,
        trim_python_function_args=args.trim_python_function_args,
        trim_python_function_name=args.trim_python_function_name,
        trim_tids=args.trim_tids,
        trim_frame_every_n=args.trim_frame_every_n,
    )
    data = _load_json(args)
    events = data["traceEvents"]

    new_events = process(events, args)
    print(f"After processing: {len(events)} -> {len(new_events)} events")
    data["traceEvents"] = new_events

    output = args.path.replace(".json", "_proced.json")
    print("*" * 100)
    print(f"Saving to {output}...")
    with open(output, "w") as f:
        json.dump(data, f, indent=None)
    return


if __name__ == "__main__":
    main()
