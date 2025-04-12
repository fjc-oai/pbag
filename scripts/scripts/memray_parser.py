import json
from memray import FileReader

import os
path = os.path.expanduser("~/Desktop/mem2/mem_profile_169320.bin")
reader = FileReader(path)

allocations = []
records = reader.get_allocation_records()
for alloc in records:
    if "TrainActor" not in alloc.thread_name:
        continue
    try:
        stack_trace = [
            [x for x in frame] for frame in alloc.stack_trace()
        ]
        allocations.append({
            "size": alloc.size,
            "tid": alloc.thread_name,
            "stack_trace": stack_trace,
        })
    except Exception as e:
        print(f"Error processing allocation: {e}")

with open("allocations.json", "w") as f:
    json.dump(allocations, f, indent=2)
print(f"dumped {len(allocations)} allocations to allocations.json")