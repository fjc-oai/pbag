"""
Torch profiler results sometimes can be tedious to read
- Irrelevant threads appearing between the main thread and cuda stream events.
- Excessively deep stack frames made cuda stream far from python functions.


Inspect original trace file to find the tids to keep, max_depth to keep.
Usage:
    python parse_torch_profiler.py /path/to/torch_profiler.json --tids=4019082,4019083 --max-depth=40

Visualization tools:
    chrome://tracing/
    https://ui.perfetto.dev/ (slower to load but much smoother to interact with!)
"""

import argparse
import json
import os
from collections import defaultdict

import tqdm


def filter_tid(events, tids_to_keep: list[int]):
    def _keep(event):
        if event.get("cat", " ") != "python_function":
            return True
        if event.get("tid", " ") in tids_to_keep:
            return True
        return False

    remained = []
    for event in tqdm.tqdm(events):
        if _keep(event):
            remained.append(event)
    return remained


def _get_main_tid(events) -> int:
    tids = defaultdict(int)
    for event in events:
        if event.get("cat", " ") == "python_function":
            tids[event["tid"]] += 1
    main_tid = max(tids, key=tids.get)
    return main_tid


def _trim_frame(events, max_depth: int) -> list:
    events = sorted(events, key=lambda e: (e["ts"], e["ts"] + e.get("dur", 0)))
    remained = []
    stack = []
    for event in tqdm.tqdm(events):
        if event.get("cat", " ") != "python_function":
            remained.append(event)
            continue

        while stack and event["ts"] >= stack[-1]["ts"] + stack[-1].get("dur", 0):
            stack.pop()

        stack.append(event)
        if (
            len(stack) <= max_depth
            or event.get("dur") is None
            or "launchkernel" in event.get("name", "").lower()
        ):
            remained.append(event)
    return remained


def _process_a_file(file: str, tids: str, max_depth: int):
    print(f"Processing {file}...")
    with open(file, "r") as f:
        data = json.load(f)
    events = data["traceEvents"]

    if tids == "auto":
        main_tid = _get_main_tid(events)
        print(f"Using main tid {main_tid} as the main thread")
        tids = [main_tid]
    else:
        tids = [int(tid) for tid in tids.split(",")]
    events = filter_tid(events, tids)

    if max_depth > 0:
        print(f"Trimming frame depth to {max_depth}...")
        events = _trim_frame(events, max_depth)

    data["traceEvents"] = events
    output = file.replace(".json", "_processed.json")
    print(f"Saving to {output}...")
    with open(output, "w") as f:
        json.dump(data, f, indent=None)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str)
    argparser.add_argument("--tids", type=str, default="auto")
    argparser.add_argument("--max-depth", type=int, default=0)
    args = argparser.parse_args()

    path = args.path
    tids = args.tids
    max_depth = args.max_depth

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    files = []
    if os.path.isfile(path):
        files.append(path)
    else:
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if (
                os.path.isfile(full_path)
                and full_path.endswith(".json")
                and "processed" not in full_path
            ):
                files.append(full_path)

    for file in files:
        _process_a_file(file, tids, max_depth)


if __name__ == "__main__":
    main()
