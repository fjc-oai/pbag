import torch
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

# Step 1: Construct the computation graph
x = torch.randn(4, 5, requires_grad=True)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# Perform computations
y = x @ w        # Element-wise multiplication
z = y + b        # Add bias
z.retain_grad()
output = z.sum() # Sum all elements to get a scalar output
output.retain_grad()

# Step 2: Visualize the computation graph using torchviz
dot = make_dot(output, params={"x": x, "w": w, "b": b})
dot.format = 'pdf'
dot.render('computation_graph')

# Step 3: Inspect the computation graph
print("output.grad_fn:", output.grad_fn)
print("output.grad_fn.next_functions:", output.grad_fn.next_functions)
print()

def print_graph(fn, indent=0):
    if fn is None:
        return
    print(' ' * indent + f'-> {type(fn).__name__}')
    for next_fn, _ in fn.next_functions:
        print_graph(next_fn, indent + 4)

print("Computation Graph:")
print_graph(output.grad_fn)

# Perform backward pass
output.backward()
# breakpoint()

# Inspect gradients
print("\nGradients:")
print(f"x.grad: {x.grad}")
print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")

# # Alternative visualization using TensorBoard
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#     def forward(self, x, w, b):
#         y = x * w
#         z = y + b
#         output = z.sum()
#         return output

# model = SimpleModel()

# # Create a SummaryWriter
# writer = SummaryWriter('runs/computation_graph_example')

# # Log the graph
# writer.add_graph(model, (x, w, b))

# # Close the writer
# writer.close()