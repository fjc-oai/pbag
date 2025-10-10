- [Lab A — single-process, two-QP RC ping-pong (no sockets!)](#lab-a--single-process-two-qp-rc-ping-pong-no-sockets)
  - [major concepts](#major-concepts)
  - [lifecycle](#lifecycle)
  - [3 apis](#3-apis)
- [Lab B — cross-host RC using the reference sample](#lab-b--cross-host-rc-using-the-reference-sample)
- [Lab C — QP Across Hosts (qp\_across\_hosts.c)](#lab-c--qp-across-hosts-qp_across_hostsc)
  - [1) Buffer registration (MR) — what and why](#1-buffer-registration-mr--what-and-why)
  - [2) How peers find each other and exchange info](#2-how-peers-find-each-other-and-exchange-info)
  - [3) Moving the QP: INIT → RTR → RTS (where the peer info is used)](#3-moving-the-qp-init--rtr--rts-where-the-peer-info-is-used)
  - [4) Posting work \& polling completions](#4-posting-work--polling-completions)
  - [Quick checklist (both hosts)](#quick-checklist-both-hosts)
- [Lab D - QP Across Hosts on GPU Memory](#lab-d---qp-across-hosts-on-gpu-memory)

# Lab A — single-process, two-QP RC ping-pong (no sockets!)
```
cc -O2 -Wall qp_single_host.c -libverbs -o qp_single_host && ./qp_single_host
```
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


# Lab B — cross-host RC using the reference sample

```
sudo apt-get install -y ibverbs-utils rdma-core
ibv_devices
```

on host A
```
$ ibv_rc_pingpong -d mlx5_ib0 -i 1 -s 1024
  local address:  LID 0x3a1e, QPN 0x000fb4, PSN 0x7560c3, GID ::
  remote address: LID 0x3c22, QPN 0x01ddb9, PSN 0x0943c2, GID ::
2048000 bytes in 0.01 seconds = 1384.84 Mbit/sec
1000 iters in 0.01 seconds = 11.83 usec/iter
```

on host B
```
$ ibv_rc_pingpong -d mlx5_ib0 -i 1 -s 1024  <hostA-ip>
  local address:  LID 0x3c22, QPN 0x01ddb9, PSN 0x0943c2, GID ::
  remote address: LID 0x3a1e, QPN 0x000fb4, PSN 0x7560c3, GID ::
2048000 bytes in 0.01 seconds = 1480.44 Mbit/sec
1000 iters in 0.01 seconds = 11.07 usec/iter
```

# Lab C — QP Across Hosts (qp_across_hosts.c)

This lab brings the pieces together to run RDMA between two different hosts. The flow is:
1) create local resources → 2) register a buffer (MR) → 3) exchange connection info out-of-band (TCP) → 4) move QPs to RTR/RTS using the peer’s info → 5) issue verbs (post_send/post_recv, or RDMA read/write) → 6) poll completions.

## 1) Buffer registration (MR) — what and why
- Allocate a local buffer
  - e.g., posix_memalign / malloc for send/recv or for one-sided RDMA. It’s just regular userspace memory at this point.
- Register it with the HCA: ibv_reg_mr
  - This pins the memory and returns an MR (memory region) handle with:
  - lkey (local key): proves to the HCA that the local buffer is registered (used in SGE).
  - rkey (remote key): lets a remote peer access this region for one-sided ops (RDMA read/write/atomic).
- Use in work requests
  - For two-sided send/recv: only the local buffer address + lkey go into your SGE; the receiver has posted a matching recv into its local buffer.
  - For one-sided RDMA: you also need the peer’s (remote_addr, rkey) to fill wr.wr.rdma.remote_addr and wr.wr.rdma.rkey.

TL;DR: register local → get (addr, length, lkey, rkey); share (addr, rkey) with your peer if you’ll do RDMA to their buffer.

## 2) How peers find each other and exchange info

We use a tiny TCP bootstrap to exchange the minimum needed fields. Each process fills a struct with its local connection parameters, sends it to the other side, and receives the peer’s struct back.

Typical fields (exact names may differ in the code):

```c
struct conn_info {
  // addressing for link-layer/routing
  union ibv_gid gid;   // RoCE: required. (For IB on Infiniband, you might use LID.)
  uint16_t lid;        // IB only; 0 for RoCE.

  // queue pair identity
  uint32_t qpn;        // QP number
  uint32_t psn;        // packet sequence number

  // optional path/port metadata
  uint8_t  port_num;   // local HCA port
  uint8_t  sl;         // service level / traffic class
  uint8_t  mtu;        // path MTU

  // memory access for one-sided ops
  uint64_t raddr;      // remote buffer VA (from ibv_reg_mr’d region)
  uint32_t rkey;       // remote key
};
```

Exchange rules:
- Each side populates conn_info with its own values (after ibv_create_qp and ibv_reg_mr).
- Use a simple TCP client/server (rank 0 listens; rank 1 connects) to swap these structs.
- After the swap, you have everything needed to:
  - call ibv_modify_qp to RTR using peer qpn, psn, gid/lid, mtu, sl, etc.
  - call ibv_modify_qp to RTS (own send params).
  - fill one-sided WRs with peer (raddr, rkey) if doing RDMA read/write.

## 3) Moving the QP: INIT → RTR → RTS (where the peer info is used)
- INIT → RTR: you pass the peer addressing (RoCE: GID; IB: LID), peer QPN, peer PSN, and path attrs (MTU, SL, port). That tells your QP how to reach the other QP for receives.
- RTR → RTS: you set your send-side params (PSN, timeouts, retry counts, etc.) to start transmitting.

## 4) Posting work & polling completions
- Two-sided (SEND/RECV):
  - Receiver posts ibv_post_recv with SGE {addr, length, lkey}.
  - Sender posts ibv_post_send with opcode IBV_WR_SEND and SGE pointing at its local buffer.
  - Completions arrive in the CQ; call ibv_poll_cq() (non-blocking) until you get WCs.
- One-sided (RDMA READ/WRITE):
  - No matching recv on the remote.
  - Sender fills wr.opcode = IBV_WR_RDMA_{WRITE,READ}, sets wr.wr.rdma.remote_addr = peer.raddr, wr.wr.rdma.rkey = peer.rkey, plus local SGE.
  - Poll CQ for completion on the initiator side.

Terminology: WR (work request) is what you post; the HCA turns it into a WQE (work queue entry). ibv_poll_cq is non-blocking; it returns 0 if nothing is ready, or N completions if available.

## Quick checklist (both hosts)
1. Open device / PD / CQ.
2. Create QP (IBV_QPT_RC), transition to INIT.
3. Allocate buffer → ibv_reg_mr → record (addr, length, lkey, rkey).
4. Fill conn_info with local {gid/lid, qpn, psn, port, mtu, sl, raddr, rkey}.
5. Exchange conn_info over TCP.
6. INIT → RTR using peer addressing + QPN/PSN.
7. RTR → RTS (local send params).
8. Post recv (if two-sided), then send/rdma_write/rdma_read.
9. ibv_poll_cq() for completions; check wc.status.


# Lab D - QP Across Hosts on GPU Memory
- Everything is the same as CPU version, except to create and pass around the address of GPU buf!!