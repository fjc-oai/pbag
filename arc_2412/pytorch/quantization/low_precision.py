import torch
from tabulate import tabulate


def dtypes():
    return [torch.float32, torch.float16, torch.bfloat16]


def cmp(t1, t2):
    abs_diff = (t1 - t2).abs().max().item()
    rel_diff = ((t1 - t2) / (t1.abs() + 1e-8)).abs().max().item()
    return abs_diff, rel_diff


N = 32


def test_conversion_error():
    table = []
    t = torch.rand(N, N, dtype=torch.float32)
    for dtype in dtypes():
        t2 = t.to(dtype).to(torch.float32)
        abs_diff, rel_diff = cmp(t, t2)
        table.append([dtype, abs_diff, rel_diff])
    print("*" * 50)
    print("test_conversion_error")
    print(tabulate(table, headers=["dtype", "abs_diff", "rel_diff"]))


def test_multiply_error():
    table = []
    ref = None
    for dtype in dtypes():
        t = torch.tensor([0.7], dtype=dtype)
        for _ in range(100):
            t = t * 2 / 2.0001
        if ref is None:
            ref = t
        abs_diff, rel_diff = cmp(ref, t)
        table.append([dtype, abs_diff, rel_diff])
    print("*" * 50)
    print("test_multiply_error")
    print(tabulate(table, headers=["dtype", "abs_diff", "rel_diff"]))


def test_range():
    table = []
    for dtype in dtypes():
        t = torch.tensor([1.0], dtype=dtype)
        r_to_inf = 0
        while not t.isinf():
            r_to_inf += 1
            t = t * 2
        t = torch.tensor([1.0], dtype=dtype)
        r_to_zero = 0
        while t != 0:
            r_to_zero += 1
            t = t / 2
        table.append([dtype, r_to_inf, r_to_zero])
    print("*" * 50)
    print("test_range")
    print(tabulate(table, headers=["dtype", "r_to_inf", "r_to_zero"]))


def test_matmul_error():
    a = torch.rand(N, N, dtype=torch.float32)
    b = torch.rand(N, N, dtype=torch.float32)
    x = torch.matmul(a, b)
    x1 = torch.matmul(a.to(torch.float16).to(torch.float32), b).to(torch.float32)
    x2 = torch.matmul(a, b.to(torch.float16).to(torch.float32)).to(torch.float32)
    x3 = torch.matmul(
        a.to(torch.bfloat16).to(torch.float32), b.to(torch.float16).to(torch.float32)
    ).to(torch.float32)
    table = []
    table.append(["float32", *cmp(x, x)])
    table.append(["float16_on_a", *cmp(x, x1)])
    table.append(["float16_on_b", *cmp(x, x2)])
    table.append(["float16_on_a_b", *cmp(x, x3)])
    print("*" * 50)
    print("test_matmul_error")
    print("*" * 50)
    print(tabulate(table, headers=["conversion", "abs_diff", "rel_diff"]))


def profile_conversion():
    """
    -> aten::to
    -> aten::_to_copy
    -> aten::copy
    -> cudaLaunchKernel
    -> void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}, at::detail::Array<char*, 2>, TrivialOffsetCalculator<1, unsigned int>, char*, at::native::memory::LoadWithCast<1>, at::detail::Array<char*, 2>::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}, at::detail::Array<char*, 2>, TrivialOffsetCalculator<1, unsigned int>, char*, at::native::memory::LoadWithCast<1>, at::detail::Array<char*, 2>::StoreWithCast<1>)
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as p:
        x = torch.rand(1000, 1000, dtype=torch.float32, device="cuda")
        y = x.to(torch.float16)


def main():
    test_conversion_error()
    test_multiply_error()
    test_range()
    test_matmul_error()
    profile_conversion()


if __name__ == "__main__":
    main()
