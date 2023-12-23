from mpi4py import MPI
import numpy as np
import argparse

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def reduce():
    data = rank
    data = comm.reduce(data, op=MPI.SUM, root=0)
    print(f"Reduce {rank=}, {data=}")
    if rank == 0:
        assert data == sum(range(size))
    else:
        assert data is None


def allreduce():
    data = rank
    data = comm.allreduce(data, op=MPI.SUM)
    print(f"Allreduce {rank=}, {data=}")
    assert data == sum(range(size))


def reduce_scatter():
    data = [rank * (10**x) for x in range(size)]
    send_buf = np.array(data, dtype=np.float32)
    recv_buf = np.empty(1, dtype=np.float32)
    comm.Reduce_scatter(send_buf, recv_buf, op=MPI.SUM)
    print(f"Reducescatter {rank=}, {recv_buf=}")
    expected = sum(range(size)) * (10**rank)
    assert np.allclose(recv_buf, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="reduce")
    args = parser.parse_args()
    fn_map = {
        "reduce": reduce,
        "allreduce": allreduce,
        "reduce_scatter": reduce_scatter,
    }
    fn_map[args.mode]()


if __name__ == "__main__":
    main()
