"""
Some profile traces are too large to be visualized, while we're still interested
to see the call stack for a specific kernel event. This script provides a way to
extract the call stack for a specific kernel event by name or external id.

Usage:
    python scripts/peek_trace.py <path-to-trace> <kernel-name>

Example:
    python scripts/peek_trace.py trace.json ncclKernel_AllGather

"""

import argparse
import os
from dataclasses import dataclass, field

from slim_trace import _load_json, trim_tid_pid
from tqdm import tqdm


@dataclass
class Node:
    event: dict
    children: list["Node"] = field(default_factory=list)
    parent: "Node" = None


"""
    # Example of a cuda kernel event
    # cuda_events = [e for e in events if e['ph']=='X' and e['cat']=='kernel']
    # pp cuda_events[0]
    {'args': {'External id': 311878,
            'block': [512, 1, 1],
            'blocks per SM': 4.0,
            'context': 1,
            'correlation': 578471,
            'device': 0,
            'est. achieved occupancy %': 100,
            'grid': [264, 2, 1],
            'queued': 0,
            'registers per thread': 16,
            'shared memory': 0,
            'stream': 7,
            'warps per SM': 64.0},
    'cat': 'kernel',
    'dur': 3.36,
    'name': 'void at::native::(anonymous '
            'namespace)::CatArrayBatchedCopy<at::native::(anonymous '
            'namespace)::OpaqueType<8u>, unsigned int, 1, 64, '
            '64>(at::native::(anonymous namespace)::OpaqueType<8u>*, '
            'at::native::(anonymous '
            'namespace)::CatArrInputTensorMetadata<at::native::(anonymous '
            'namespace)::OpaqueType<8u>, unsigned int, 64, 64>, '
            'at::native::(anonymous namespace)::TensorSizeStride<unsigned int, '
            '4u>, int, unsigned int)',
    'ph': 'X',
    'pid': 0,
    'tid': 7,
    'ts': 3777862676194.833}
    }
"""


def event_key(event) -> str:
    """Creates a unique key for an event based on its timestamp and duration."""
    return f"{event['ts']}-{event['dur']}"


def filter_events(events):
    """Splits the events into different categories."""
    kernel_events = [e for e in events if e["ph"] == "X" and e["cat"] == "kernel"]
    cuda_runtime_events = [e for e in events if e["ph"] == "X" and e["cat"] == "cuda_runtime"]
    python_events = [e for e in events if e["ph"] == "X" and e["cat"] == "python_function"]
    cpu_op_events = [e for e in events if e["ph"] == "X" and e["cat"] == "cpu_op"]
    return kernel_events, cuda_runtime_events, python_events, cpu_op_events


def build_tree(events):
    """
    Build a tree structure for the events.

    Returns a dummy root node and a mapping from event key to Node.
    """
    event_to_node = {}
    events.sort(key=lambda e: e["ts"])
    dummy_event = {"ts": float("-inf"), "dur": float("inf"), "name": "dummy"}
    root_node = Node(dummy_event)
    stack = [root_node]

    for event in tqdm(events, desc="Building tree"):
        node = Node(event)
        event_to_node[event_key(event)] = node

        # Pop until the current event is within the time window of the node at the stack's top.
        while (
            event["ts"] >= stack[-1].event["ts"] + stack[-1].event["dur"]
            or event["ts"] + event["dur"] >= stack[-1].event["ts"] + stack[-1].event["dur"]
        ):
            stack.pop()

        stack[-1].children.append(node)
        node.parent = stack[-1]
        stack.append(node)

    return root_node, event_to_node


def get_ancestors(kernel_event, cuda_runtime_events, event_to_node):
    """
    Get the list of ancestor nodes for a given kernel event.
    It first finds the corresponding cuda_runtime event based on a shared correlation value.
    """
    correlation = kernel_event["args"]["correlation"]
    # Find the corresponding cuda_runtime event matching the correlation.
    cuda_runtime_event = next(
        e
        for e in cuda_runtime_events
        if "correlation" in e["args"] and e["args"]["correlation"] == correlation
    )
    node = event_to_node[event_key(cuda_runtime_event)]
    ancestors = []
    while node:
        ancestors.append(node)
        node = node.parent
    return ancestors


def get_ancestor_names(kernel_event, cuda_runtime_events, event_to_node):
    """Extracts the names of the events in the ancestor chain."""
    ancestors = get_ancestors(kernel_event, cuda_runtime_events, event_to_node)
    return tuple(node.event["name"] for node in ancestors)


def peek_at_kernel_by_external_id(eid, kernel_events, cuda_runtime_events, event_to_node):
    """Finds a kernel event by external id and prints its ancestor call stack."""
    kernel_event = next(
        e
        for e in kernel_events
        if "args" in e and "External id" in e["args"] and e["args"]["External id"] == eid
    )
    ancestors = get_ancestors(kernel_event, cuda_runtime_events, event_to_node)
    for idx, node in enumerate(ancestors):
        print(f"{idx}: {node.event['name']}")
    return ancestors


def peek_at_kernel_by_name(name, kernel_events, cuda_runtime_events, event_to_node):
    """
    Finds kernel events whose names contain the specified substring and prints unique ancestor paths.
    """
    selected_kernel_events = [e for e in kernel_events if name in e["name"]]
    if not selected_kernel_events:
        print(f"No kernel events found with name containing '{name}'")
        return
    found = set()
    for kernel_event in selected_kernel_events:
        try:
            ancestor_names = get_ancestor_names(kernel_event, cuda_runtime_events, event_to_node)
            if ancestor_names not in found:
                print(f"Found unique ancestor path (unique index: {len(found)}):")
                for idx, ancestor_name in enumerate(ancestor_names):
                    print(f"    {idx}: {ancestor_name}")
                found.add(ancestor_names)
        except Exception as e:
            print(f"Error processing kernel event: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Parse a trace JSON file and display the call stack for a given kernel event."
    )
    parser.add_argument("file", help="Path to the JSON trace file")
    parser.add_argument("kernel", help="Kernel name (or substring) to search for")
    args = parser.parse_args()

    # Load JSON file.
    path = os.path.expanduser(args.file)
    data = _load_json(path, fix=True)
    events = data["traceEvents"]

    events = trim_tid_pid(events, tids="auto")

    # Filter events.
    kernel_events, cuda_runtime_events, python_events, cpu_op_events = filter_events(events)
    print(f"kernel_events: {len(kernel_events)}")
    print(f"cuda_runtime_events: {len(cuda_runtime_events)}")
    print(f"python_events: {len(python_events)}")
    print(f"cpu_op_events: {len(cpu_op_events)}")

    # Build the tree from interesting events.
    interesting_events = cuda_runtime_events + python_events  # + cpu_op_events
    _, event_to_node = build_tree(interesting_events)

    # Display the ancestor call stack for the specified kernel.
    peek_at_kernel_by_name(args.kernel, kernel_events, cuda_runtime_events, event_to_node)


if __name__ == "__main__":
    main()
