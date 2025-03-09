import time

import cprofiler
import yappi


def a():
    import math
    import random

    # Define matrix dimension and create two random matrices
    size = 100
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]

    # Multiply matrices A and B using triple nested loops
    result = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += A[i][k] * B[k][j]

    # Compute an aggregate value: sum of square roots of all elements in the result matrix
    aggregate_sum = 0.0
    for row in result:
        for value in row:
            aggregate_sum += math.sqrt(value)
    
    print("Aggregate sum of square roots:", aggregate_sum)

# yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
# yappi.start()

cprofiler.set_profile()

st = time.perf_counter()
a()
et = time.perf_counter()
print(f"Time taken: {et - st} seconds")

# yappi.get_func_stats().print_all()
# yappi.get_thread_stats().print_all()

cprofiler.unset_profile()

'''

Clock type: CPU
Ordered by: totaltime, desc

name                                  ncall  tsub      ttot      tavg      
doc.py:5 a                            1      0.117907  0.117907  0.117907

name           id     tid              ttot      scnt        
_MainThread    0      139867147315008  0.118297  1
'''