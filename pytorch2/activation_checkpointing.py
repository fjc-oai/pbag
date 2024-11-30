import argparse

import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


# Define a simple model with multiple layers
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.mlp = nn.Linear(1024, 1024)

    def forward(self, x):
        x = torch.relu(self.mlp(x))
        return x


class CheckpointedModel(nn.Module):
    def __init__(self):
        super(CheckpointedModel, self).__init__()
        self.mlp = nn.Linear(1024, 1024)

    def forward(self, x):
        x = checkpoint.checkpoint(self.mlp, x)
        x = checkpoint.checkpoint(torch.relu, x)
        return x


# Helper function to measure memory usage
def print_memory_usage():
    print(f"Allocated memory: {cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {cuda.memory_reserved() / 1024**2:.2f} MB\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Toggle activation checkpointing.")
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="Use activation checkpointing"
    )
    args = parser.parse_args()

    # Initialize models and data
    device = "cuda" if cuda.is_available() else "cpu"
    input_data = torch.randn(1024*1024, 1024, device=device, requires_grad=True)  # Batch size of 16

    # Start memory recording
    torch.cuda.memory._record_memory_history()

    # Select model based on the use_checkpoint flag
    if args.use_checkpoint:
        print("Running with checkpointing:")
        model = CheckpointedModel().to(device)
    else:
        print("Running without checkpointing:")
        model = SimpleModel().to(device)

    print_memory_usage()  # Check memory before
    output = model(input_data)
    output.mean().backward()  # Backward pass
    print_memory_usage()  # Check memory after

    # Dump memory snapshot
    torch.cuda.memory._dump_snapshot(f"my_snapshot_{args.use_checkpoint}.pickle")


if __name__ == "__main__":
    main()
