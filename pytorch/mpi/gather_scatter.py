from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def broadcast():
    if rank == 0:
        data = 7
    else:
        data = None
    data = comm.bcast(data, root=0)
    print(f"Broadcast {rank=}, {data=}")
    assert data == 7


def scatter():
    if rank == 0:
        data = list(range(size))
    else:
        data = None
    data = comm.scatter(data, root=0)
    print(f"Scatter {rank=}, {data=}")
    assert data == rank


def gather():
    data = rank
    data = comm.gather(data, root=0)
    print(f"Gather {rank=}, {data=}")
    if rank == 0:
        assert data == list(range(size))
    else:
        assert data is None


def allgather():
    data = rank
    data = comm.allgather(data)
    print(f"Allgather {rank=}, {data=}")
    assert data == list(range(size))


def all2all():
    data = [x * (10**rank) for x in range(size)]
    data = comm.alltoall(data)
    print(f"Alltoall {rank=}, {data=}")
    assert data == [rank * (10**x) for x in range(size)]


def gather_scatter():
    return all2all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="broadcast")
    fn_map = {
        "broadcast": broadcast,
        "scatter": scatter,
        "gather": gather,
        "allgather": allgather,
        "all2all": all2all,
        "gather_scatter": gather_scatter,
    }
    args = parser.parse_args()
    fn = fn_map[args.mode]
    fn()
