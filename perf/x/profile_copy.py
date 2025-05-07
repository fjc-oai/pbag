import torch
from torch.profiler import record_function


def fn():
    t_h = torch.tensor([1, 2, 3], device="cuda")
    t_d = torch.tensor([1, 2, 3], device="cpu", pin_memory=True)

    a = torch.randn(128, 1024, 1024, device="cuda")
    b = torch.randn(128, 1024, 1024, device="cuda")
    _ = a @ b
        
    with record_function("h2d"):
        t_d_h = t_d.to(device="cuda", non_blocking=True)

    with record_function("d2h"):
        t_h_d = t_h.to(device="cpu", non_blocking=True)

    _ = a @ b


def main():
    for _ in range(3):
        fn()

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    )
    prof.start()
    fn()
    prof.stop()
    prof.export_chrome_trace("trace.json")
    print(f"saved trace to trace.json")


if __name__ == "__main__":
    main()
