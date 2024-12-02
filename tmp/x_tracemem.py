import tracemalloc

import torch

INCLUDE_FREE = False

tracemalloc.start(20)
t1 = torch.rand(1000, 1000)
t2 = torch.rand(1000, 1000)
t3 = torch.rand(1000, 1000)


s1 = tracemalloc.take_snapshot()

del t3
t4 = torch.rand(1000, 1000)
t5 = torch.rand(1000, 1000)
t6 = torch.rand(1000, 1000)

s2 = tracemalloc.take_snapshot()

stats = s2.compare_to(s1, 'filename')
breakpoint()


