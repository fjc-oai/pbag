## major concepts

 - device / port / GID: device is the HCA function, port is its physical interface, and GID is the logical address on that port.
 - PD (Protection Domain): hardware-enforced isolation domain; QPs and MRs must be in the same PD to interact.
 - MR (Memory Region): registered and pinned memory range tied to a PD, yielding lkey/rkey for DMA access.
 - SGE: buffer descriptor used in WRs that carries address, length, and lkey; multiple SGEs can be chained.
 - CQ (Completion Queue): queue where completed work generates CQEs that you poll or wait on.
 - QP (Queue Pair): endpoint with a send queue and receive queue that transitions RESET→INIT→RTR→RTS.
 - WR / CQE: a WR requests work (send/recv/rdma/atomic) on a queue, and a CQE reports its completion status.

```
HCA / device → port → GID/LID → context (ibv_open_device)
      → PD → MR ↔ SGE
      → CQ → QP (SQ/RQ)
      → WRs (SEND/RECV) → CQEs (ibv_poll_cq)
```

## lifecycle

- discover hardware; query port and GID
- create isolation by allocating a Protection Domain
- create CQ
- register MRs (lkey/rkey)
- create QPs (bind to CQ); print QP info
- states: RESET → INIT → RTR → RTS
- two-sided: pre-post RECV; post SEND
- completions: poll CQ; verify data

## 3 apis

- ibv_post_recv(qp, wr, &bad): enqueue receive WQEs (two-sided only); keep a backlog
- ibv_post_send(qp, wr, &bad): enqueue SEND/RDMA/ATOMIC WQEs on the send queue
- ibv_poll_cq(cq, n, wc): non-blocking; returns 0 if no CQEs; <0 on error; for blocking use a completion channel

Modes (how they combine):
- recv only: two-sided receive path; post RECVs; remote posts SEND; you see RECV CQEs.
- send only: one-sided RDMA WRITE/READ; post SEND (RDMA_*); remote posts no RECV; only initiator sees CQEs.
- send and recv: two-sided messaging; receiver pre-posts RECV; sender posts SEND; both sides get CQEs.


