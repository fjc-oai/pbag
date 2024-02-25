"""
Torch profiler results sometimes can be tedious to read, due to irrelevant 
threads appearing between the main thread and cuda stream events. And it 
makes chrome://tracing super slow!!!

Usage:
    Inspect original trace file to find the tids to keep. 
    Then run the script with the tids as the second argument.
    python parse_torch_profiler.py /path/to/torch_profiler.json 4019082,4019083

Visualization tools:
    chrome://tracing/
    https://ui.perfetto.dev/
"""

import argparse
import json
import os

import tqdm


def clean_up(input_file, output_file, tids_to_keep=None):
    with open(input_file, "r") as f:
        data = json.load(f)

    def _keep(event):
        if event.get("cat", " ") != "python_function":
            return True
        if event.get("tid", " ") in tids_to_keep:
            return True
        return False

    filtered_events = []
    for event in tqdm.tqdm(data["traceEvents"]):
        if _keep(event):
            filtered_events.append(event)
    data["traceEvents"] = filtered_events

    with open(output_file, "w") as f:
        json.dump(data, f, indent=None)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str)
    argparser.add_argument("tids", type=str)
    args = argparser.parse_args()

    input_file = os.path.expanduser(args.path)
    output_file = args.path.replace(".json", "_filtered.json")
    tids = [int(tid) for tid in args.tids.split(",")]
    clean_up(input_file, output_file, tids)


if __name__ == "__main__":
    main()
