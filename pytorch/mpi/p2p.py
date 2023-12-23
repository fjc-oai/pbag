# https://mpi4py.readthedocs.io/en/stable/tutorial.html#running-python-scripts-with-mpi
#
# mpiexec -n 2 python p2p.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def sync():
    if rank == 0:
        data = np.arange(10, dtype=np.float32)
        comm.Send(data, dest=1, tag=11)
    elif rank == 1:
        data = np.empty(10, dtype=np.float32)
        comm.Recv(data, source=0, tag=11)
    print(f"{rank=}, {data=}")
    assert np.allclose(data, np.arange(10, dtype=np.float32))


def async_():
    if rank == 0:
        data = np.arange(10, dtype=np.float32)
        req = comm.Isend(data, dest=1, tag=11)
        req.Wait()
    elif rank == 1:
        data = np.empty(10, dtype=np.float32)
        req = comm.Irecv(data, source=0, tag=11)
        req.Wait()
    print(f"{rank=}, {data=}")
    assert np.allclose(data, np.arange(10, dtype=np.float32))


def buf():
    if rank == 0:
        data = np.array([7], dtype=np.float32)
        comm.Send([data, MPI.FLOAT], dest=1, tag=11)
    elif rank == 1:
        data = np.empty(1, dtype=np.float32)
        comm.Recv([data, MPI.FLOAT], source=0, tag=11)
    print(f"{rank=}, {data=}")
    assert np.allclose(data, np.array([7], dtype=np.float32))


def main():
    sync()
    async_()
    buf()


if __name__ == "__main__":
    main()
