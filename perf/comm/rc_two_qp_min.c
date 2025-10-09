// cc -O2 -Wall rc_two_qp_min.c -libverbs -o rc_two_qp_min && ./rc_two_qp_min
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

static const char* node_type_str(enum ibv_node_type t) {
    switch (t) {
        case IBV_NODE_CA: return "CA";
        case IBV_NODE_SWITCH: return "SWITCH";
        case IBV_NODE_ROUTER: return "ROUTER";
        case IBV_NODE_RNIC: return "RNIC";
        case IBV_NODE_USNIC: return "USNIC";
        case IBV_NODE_USNIC_UDP: return "USNIC_UDP";
        default: return "UNKNOWN";
    }
}

static const char* transport_type_str(enum ibv_transport_type t) {
    switch (t) {
        case IBV_TRANSPORT_IB: return "IB/RoCE";
        case IBV_TRANSPORT_IWARP: return "iWARP";
        case IBV_TRANSPORT_USNIC: return "usNIC";
        case IBV_TRANSPORT_USNIC_UDP: return "usNIC_UDP";
        default: return "UNKNOWN";
    }
}

static void print_gid(const union ibv_gid *gid) {
    // Print GID as 16-byte hex
    for (int i = 0; i < 16; i++) {
        printf("%02x%s", gid->raw[i], (i == 15) ? "" : "");
    }
}

#define CHECK(x) do { if ((x)) { fprintf(stderr, "ERR %s:%d: %s\n", __FILE__, __LINE__, strerror(errno)); exit(1);} } while (0)
#define CEQ(x,msg) do { if ((x)) { fprintf(stderr, "ERR %s:%d: %s\n", __FILE__, __LINE__, msg); exit(1);} } while (0)

static uint32_t rand32() { return (uint32_t) (rand() & 0x7fffffff); }

// Step banner helper
static void print_step_banner(const char* message) {
    const int totalWidth = 80; // total characters for the banner line
    if (!message) message = "";
    int messageLength = (int)strlen(message);
    int minStarsEachSide = 2; // ensure visibility even for long messages
    int availableStars = totalWidth - (messageLength + 2); // spaces around message
    if (availableStars < minStarsEachSide * 2) {
        availableStars = minStarsEachSide * 2;
    }
    int starsLeft = availableStars / 2;
    int starsRight = availableStars - starsLeft;

    putchar('\n');
    for (int i = 0; i < starsLeft; ++i) putchar('*');
    putchar(' ');
    fputs(message, stdout);
    putchar(' ');
    for (int i = 0; i < starsRight; ++i) putchar('*');
    putchar('\n');
    fflush(stdout);
}
#define STEP(msg) do { print_step_banner((msg)); } while (0)

static const char* qp_type_str(enum ibv_qp_type t) {
    switch (t) {
        case IBV_QPT_RC: return "RC";
        case IBV_QPT_UC: return "UC";
        case IBV_QPT_UD: return "UD";
#ifdef IBV_QPT_RAW_PACKET
        case IBV_QPT_RAW_PACKET: return "RAW_PACKET";
#endif
#ifdef IBV_QPT_DRIVER
        case IBV_QPT_DRIVER: return "DRIVER";
#endif
        default: return "UNKNOWN";
    }
}

static const char* qp_state_str(enum ibv_qp_state s) {
    switch (s) {
        case IBV_QPS_RESET: return "RESET";
        case IBV_QPS_INIT:  return "INIT";
        case IBV_QPS_RTR:   return "RTR";
        case IBV_QPS_RTS:   return "RTS";
        case IBV_QPS_SQD:   return "SQD";
        case IBV_QPS_SQE:   return "SQE";
        case IBV_QPS_ERR:   return "ERR";
        default: return "UNKNOWN";
    }
}

static void print_qp_info(struct ibv_qp* qp, const char* label) {
    if (!qp) {
        printf("QP %s: (null)\n", label ? label : "?");
        return;
    }
    printf("QP %s: ptr=%p qp_num=%u qp_type=%s pd=%p send_cq=%p recv_cq=%p\n",
           label ? label : "?",
           (void*)qp, qp->qp_num, qp_type_str(qp->qp_type),
           (void*)qp->pd, (void*)qp->send_cq, (void*)qp->recv_cq);

    struct ibv_qp_attr attr; memset(&attr, 0, sizeof(attr));
    struct ibv_qp_init_attr init_attr; memset(&init_attr, 0, sizeof(init_attr));
    int r = ibv_query_qp(qp, &attr, IBV_QP_STATE | IBV_QP_CAP, &init_attr);
    if (r == 0) {
        printf("  state=%s cap(send_wr=%u recv_wr=%u send_sge=%u recv_sge=%u inline=%u)\n",
               qp_state_str(attr.qp_state),
               init_attr.cap.max_send_wr, init_attr.cap.max_recv_wr,
               init_attr.cap.max_send_sge, init_attr.cap.max_recv_sge,
               init_attr.cap.max_inline_data);
    } else {
        printf("  ibv_query_qp failed: %s\n", strerror(errno));
    }
}


static void qp_to_init(struct ibv_qp* qp, uint8_t port) {
    struct ibv_qp_attr a = {0};
    a.qp_state        = IBV_QPS_INIT;
    a.pkey_index      = 0;
    a.port_num        = port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_READ  |
                        IBV_ACCESS_REMOTE_WRITE;
    int r = ibv_modify_qp(qp, &a,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    CEQ(r, "modify_qp INIT failed");
}

static void qp_to_rtr(struct ibv_qp* qp, uint32_t dest_qpn, uint16_t dlid,
                      union ibv_gid dgid, uint8_t port, uint8_t mtu, uint32_t rq_psn) {
    struct ibv_qp_attr a = {0};
    a.qp_state           = IBV_QPS_RTR;
    a.path_mtu           = mtu;          // IBV_MTU_1024 etc.
    a.dest_qp_num        = dest_qpn;
    a.rq_psn             = rq_psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer      = 12;

    a.ah_attr.is_global  = 1;
    a.ah_attr.grh.dgid   = dgid;
    a.ah_attr.grh.hop_limit = 1;
    a.ah_attr.grh.sgid_index = 0;
    a.ah_attr.dlid       = dlid;
    a.ah_attr.sl         = 0;
    a.ah_attr.src_path_bits = 0;
    a.ah_attr.port_num   = port;

    int r = ibv_modify_qp(qp, &a,
        IBV_QP_STATE      | IBV_QP_AV | IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN   | IBV_QP_RQ_PSN |
        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    CEQ(r, "modify_qp RTR failed");
}

static void qp_to_rts(struct ibv_qp* qp, uint32_t sq_psn) {
    struct ibv_qp_attr a = {0};
    a.qp_state      = IBV_QPS_RTS;
    a.timeout       = 14;
    a.retry_cnt     = 7;
    a.rnr_retry     = 7;     // infinite RNR retry
    a.sq_psn        = sq_psn;
    a.max_rd_atomic = 1;
    int r = ibv_modify_qp(qp, &a,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    CEQ(r, "modify_qp RTS failed");
}

int main() {
    srand(time(NULL));
    int num_dev = 0;
    STEP("get device list");
    struct ibv_device **devs = ibv_get_device_list(&num_dev);
    CEQ(!devs || num_dev == 0, "no verbs devices");
    // Verbs device enumeration:
    // - ibv_get_device_list() returns RDMA-capable devices visible to libibverbs.
    // - Names like "mlx5_ibX" are real HCA functions usable for verbs; we can open them.
    // - Names like "mlx5_anX" are auxiliary representors and are not usable as verbs devices,
    //   so we skip them when choosing which device to open.
    // - GUID is a stable EUI-64-style identifier for the device/port function (0 often means
    //   an aux/non-RDMA entity).
    // - HCA (Host Channel Adapter) is the RDMA NIC. It exposes queue pairs (QPs), completion
    //   queues (CQs), protection domains (PDs), and memory registration (MR) to the verbs API,
    //   and offloads RDMA operations (SEND/RECV, RDMA READ/WRITE, atomics) on your behalf.

    /*
        num_dev: 6
        dev[0]: name=mlx5_an0, guid=0000000000000000
        dev[1]: name=mlx5_an1, guid=70630bfeff521e7e
        dev[2]: name=mlx5_ib0, guid=9b0f34feff5d1500
        dev[3]: name=mlx5_ib1, guid=9c0f34feff5d1500
        dev[4]: name=mlx5_ib2, guid=9d0f34feff5d1500
        dev[5]: name=mlx5_ib3, guid=9e0f34feff5d1500
    */
    printf("num_dev: %d\n", num_dev);
    for (int i = 0; i < num_dev; i++) {
        const char *name = ibv_get_device_name(devs[i]);
        uint64_t guid = ibv_get_device_guid(devs[i]);
        printf("  dev[%d]: name=%s, guid=%016" PRIx64 "\n", i, name ? name : "(null)", guid);
    }
    int chosen_idx = -1;
    for (int i = 0; i < num_dev; i++) {
        const char *name = ibv_get_device_name(devs[i]);
        if (name && strncmp(name, "mlx5_an", 7) == 0) {
            continue; // skip auxiliary devices
        }
        
        chosen_idx = i;
        if (name && strncmp(name, "mlx5_ib", 7) == 0) {
            break; // prefer mlx5_ib explicitly if present
        }
    }
    CEQ(chosen_idx < 0, "no suitable RDMA device (skipped aux 'mlx5_an')");
    STEP("open device");
    printf("opening dev[%d]: %s\n", chosen_idx, ibv_get_device_name(devs[chosen_idx]));
    struct ibv_context *ctx = ibv_open_device(devs[chosen_idx]);
    CEQ(!ctx, "open device failed");
    // ibv_context basics (process's handle to the opened RDMA device):
    // - async_fd: a pollable FD for asynchronous events (e.g., port active/down, CQ overrun,
    //   path migration, device fatal errors). You can epoll/select on this to drain events.
    // - num_comp_vectors: number of completion interrupt vectors provided by the device/driver;
    //   informs how many independent CQ event channels you can spread across for scalability.
    // - device: back-reference to the ibv_device we opened; from it you can read
    //   node_type (e.g., CA = Channel Adapter/HCA, i.e., the RDMA NIC) and transport_type
    //   (IB/RoCE, iWARP, etc.).
    
    /*
        opening dev[2]: mlx5_ib0
        ctx: async_fd=4, num_comp_vectors=19
        ctx->device: name=mlx5_ib0, guid=9b0f34feff5d1500, node_type=CA(1), transport_type=IB/RoCE(0)
    */
    printf("ctx: async_fd=%d, num_comp_vectors=%d\n", ctx->async_fd, ctx->num_comp_vectors);
    struct ibv_device *ctx_dev = ctx->device;
    if (ctx_dev) {
        const char *dname = ibv_get_device_name(ctx_dev);
        uint64_t dguid = ibv_get_device_guid(ctx_dev);
        printf("ctx->device: name=%s, guid=%016" PRIx64 ", node_type=%s(%d), transport_type=%s(%d)\n",
               dname ? dname : "(null)", dguid,
               node_type_str(ctx_dev->node_type), ctx_dev->node_type,
               transport_type_str(ctx_dev->transport_type), ctx_dev->transport_type);
    }

    /*
       Great question — in InfiniBand and RDMA, “port” is primarily a physical concept, but it also carries a logical role.

       Here’s the breakdown:

       ⸻

       🧱 1. Physical meaning of a port
           •   A “port” corresponds to a real network interface on the HCA (Host Channel Adapter / NIC).
           •   A dual-port NIC literally has two physical QSFP connectors, each one a port.
           •   Each physical port:
           •   Has its own LID, GID table, and link state (LINK_UP / LINK_DOWN).
           •   Connects to the fabric through a cable → switch or peer node.
           •   Can be on different subnets or fabrics.

       So at the lowest level, port = physical connector/interface.

       ⸻

       🧠 2. Logical aspects of a port
           •   Each physical port can be configured with multiple GIDs (think multiple IP addresses on one NIC).
           •   Each port can carry multiple Queue Pairs (QPs), each belonging to a different application or connection.
           •   The verbs API exposes ports as logical endpoints you can bind QPs to.
           •   Example:

       ibv_query_port(ctx, port_num, &port_attr);
       ibv_query_gid(ctx, port_num, gid_index, &gid);

       — here port_num selects the port both physically and logically.

       ⸻

       📶 3. Analogy with Ethernet
           •   Think of port like:
           •   eth0, eth1 = physical interfaces
           •   each can have multiple IP addresses (logical GIDs)
           •   sockets bind to an IP on a specific interface.

       InfiniBand works similarly:
           •   HCA → device
           •   Port → physical connector + logical interface
           •   GID → address on that port
    */
    STEP("query port attributes");
    uint8_t port = 1;
    struct ibv_port_attr port_attr;
    CEQ(ibv_query_port(ctx, port, &port_attr), "query_port failed");
    STEP("query gid (sgid_index=0)");
    union ibv_gid gid = {0};
    CEQ(ibv_query_gid(ctx, port, 0, &gid), "query_gid failed");
    printf("using port=%u, gid=", port);
    print_gid(&gid);
    printf("\n");

    /*
       In RDMA / InfiniBand, PD stands for Protection Domain.
       It’s a logical isolation boundary that controls which resources (QPs, MRs, CQs, etc.) can safely interact with each other.

       ⸻

       🧱 1. What a PD does

       Think of PD like a “sandbox” for RDMA resources:
           •   Every Queue Pair (QP) must belong to exactly one PD.
           •   Every Memory Region (MR) is registered with exactly one PD.
           •   RDMA operations are only valid if both the QP and MR belong to the same PD.

       This is a hardware-enforced security and isolation mechanism:

       If you try to use a QP from PD A to access memory registered in PD B, the NIC will reject the operation (local protection error).

       ⸻

       🧠 2. Why PD exists
           •   RDMA allows user-space applications to talk directly to NIC hardware (bypassing the kernel).
           •   To prevent one application from accessing another app’s memory, PDs act like “namespaces” or “fences”.
           •   Typically:
           •   One PD per application/process.
           •   All QPs and MRs in that process belong to the same PD.
           •   Multiple PDs can exist on the same device simultaneously.
    */
    STEP("alloc PD");
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    CEQ(!pd, "alloc_pd failed");

    /*
       In RDMA / InfiniBand, CQ stands for Completion Queue.
       It’s one of the core objects in the verbs API — used to get notifications about completed work (e.g. sends, receives, RDMA reads/writes).

       ⸻

       🧠 1. What a CQ does
           •   When you post a Work Request (WR) to a Queue Pair (QP) (e.g., ibv_post_send() or ibv_post_recv()), it doesn’t complete immediately.
           •   The NIC executes the operation asynchronously.
           •   When the operation finishes (successfully or with an error), the NIC pushes a Completion Queue Entry (CQE) into the CQ.
           •   Your program polls or waits on the CQ to know when things are done.

       Think of CQ like “epoll” for RDMA — you don’t block on each send/recv; you check completions in a queue.

       ⸻

       🧱 2. CQ is shared between QPs
           •   You can associate multiple QPs with the same CQ.
           •   Each QP can have:
           •   one CQ for both send and receive completions
           •   or separate send CQ and receive CQ
           •   This design helps reduce overhead:
           •   e.g. a server with 1000 connections might have 1000 QPs but only a few shared CQs.

       ✅ TL;DR
           •   CQ = Completion Queue — NIC uses it to tell you “this work is done.”
           •   You poll or wait on it to find out when WRs finish.
           •   One CQ can serve many QPs.
           •   It’s central to making RDMA communication asynchronous and fast.
    */
    STEP("create CQ");
    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    CEQ(!cq, "create_cq failed");

    const size_t SZ = 4096;
    void *buf0 = aligned_alloc(4096, SZ);
    void *buf1 = aligned_alloc(4096, SZ);
    snprintf((char*)buf0, SZ, "hello-from-qp0");
    memset(buf1, 0, SZ);

    /*
       In RDMA / InfiniBand, MR stands for Memory Region.
       This is one of the core objects in the verbs API, and MR flags are options that describe how that memory can be accessed.

       ⸻

       🧠 1. What is an MR?

       Normally, user-space memory is invisible to the NIC.
       To let the NIC directly read or write memory buffers (zero-copy RDMA), you must register those buffers with the NIC. This registration creates a Memory Region (MR).

       When you register, the NIC:
           • Pins those pages in host memory (so they won’t be swapped out),
           • Creates metadata (like lkey/rkey) so RDMA operations can reference it,
           • Associates it with a Protection Domain (PD).

       struct ibv_mr *mr = ibv_reg_mr(pd, buf, size, access_flags);


       ⸻

       📜 2. MR flags (access flags)

       These are passed as access_flags in ibv_reg_mr(). Common ones:

       Flag	Meaning
       IBV_ACCESS_LOCAL_WRITE	Required if the local CPU will modify the buffer after registration.
       IBV_ACCESS_REMOTE_WRITE	Allow remote peers to write to this memory using RDMA Write.
       IBV_ACCESS_REMOTE_READ	Allow remote peers to read from this memory using RDMA Read.
       IBV_ACCESS_REMOTE_ATOMIC	Allow remote atomic operations (e.g. compare-and-swap).
       IBV_ACCESS_MW_BIND	Allow the region to be used for binding memory windows.
       IBV_ACCESS_ON_DEMAND	Used for on-demand paging (ODP), advanced use.

       Example:

       mr = ibv_reg_mr(pd, buf, size,
                       IBV_ACCESS_LOCAL_WRITE |
                       IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE);

       This lets the local CPU write to the buffer, and remote QPs read/write to it.

       ⸻

       🧭 3. Why this matters
           • Security: If a remote QP doesn’t have rkey or the MR doesn’t have the right flags, the operation fails.
           • Performance: MRs enable zero-copy RDMA — NIC can DMA directly into those buffers without CPU involvement.
           • Isolation: MR is tied to a PD. Only QPs in the same PD can access it.

       ⸻

       📌 4. MR vs GPU memory

       By default, MR refers to host memory.
       But:
           • If the NIC and GPU support GPUDirect RDMA (e.g., Mellanox + NVIDIA), you can register GPU memory as an MR too.
           • That requires special setup (cudaMalloc + ibv_reg_mr() on the GPU pointer).
           • If you use aligned_alloc() or malloc(), that’s just host memory.
    */
    STEP("register MRs");
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    struct ibv_mr *mr0 = ibv_reg_mr(pd, buf0, SZ, mr_flags);
    struct ibv_mr *mr1 = ibv_reg_mr(pd, buf1, SZ, mr_flags);
    CEQ(!mr0 || !mr1, "reg_mr failed");

    /*
       What is a QP (Queue Pair)?

       A QP = Queue Pair is the core abstraction for sending and receiving RDMA messages.
       It consists of two hardware-managed queues:
           • Send Queue (SQ) — where you post outbound work requests (e.g., RDMA Write, RDMA Read, Send).
           • Receive Queue (RQ) — where you post buffers to receive incoming messages.

       Each QP:
           • Belongs to a Protection Domain (PD).
           • Is bound to specific Completion Queues (CQ) to report completions.
           • Has its own state machine (RESET → INIT → RTR → RTS, etc.).
           • Usually pairs with another QP on a remote host to form a connection.
    */
    /*
       🧱 1. What qp_type actually controls

       qp_type (e.g. IBV_QPT_RC, IBV_QPT_UC, IBV_QPT_UD, etc.) controls the transport semantics of the QP:

       QP Type            Name                   Characteristics
       IBV_QPT_RC         Reliable Connected     reliable, bidirectional, supports RDMA Read/Write, Send/Recv
       IBV_QPT_UC         Unreliable Connected   no retries, no RDMA Read, only Write and Send
       IBV_QPT_UD         Unreliable Datagram    connectionless, only Send/Recv, no RDMA Read/Write
       IBV_QPT_RAW_PACKET Raw Packet             bypass IB stack, raw Ethernet frames

       So qp_type sets what operations are allowed on that QP — but not whether local or remote participates in each operation.

       For example:
           • RC QP → you can do RDMA Read, Write, Send/Recv
           • UC QP → you can do Write and Send/Recv (no Read)
           • UD QP → you can only do Send/Recv
    */
    struct ibv_qp_init_attr qia = {
        .send_cq = cq, .recv_cq = cq,
        .cap = {.max_send_wr = 16, .max_recv_wr = 16, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0},
        .qp_type = IBV_QPT_RC
    };
    STEP("create QPs");
    struct ibv_qp *qp0 = ibv_create_qp(pd, &qia);
    struct ibv_qp *qp1 = ibv_create_qp(pd, &qia);
    CEQ(!qp0 || !qp1, "create_qp failed");
    // Print interesting fields from the QP objects
    print_qp_info(qp0, "qp0");
    print_qp_info(qp1, "qp1");

    /*
        There’s no standard Ierbs API function literally called qp_to_init() — but in many RDMA
        examples, qp_to_init() is a helper function people write themselves to:

        🧠 transition a QP from the RESET state to the INIT state.

        ⸻

        🧭 Why it exists
        • Every RDMA Queue Pair (QP) starts in the RESET state when you create it.
        • Before you can actually post any send/recv or connect to a remote peer, you must move it
            through a state machine:
            RESET → INIT → RTR (Ready to Receive) → RTS (Ready to Send)
        • Doing this requires calling:
            ibv_modify_qp(qp, &attr, attr_mask);
            with the right attributes for each transition.

        ⸻

        🧱 What qp_to_init() typically does
        • qp_state = IBV_QPS_INIT → target state
        • port_num → which HCA port the QP is bound to
        • qp_access_flags → what kinds of RDMA ops are allowed on this QP (e.g., read/write)

        📜 Related transitions
        • qp_to_rtr() → INIT → RTR
        • qp_to_rts() → RTR → RTS

        You often see these 3 helpers together in sample RDMA programs and tutorials.
    */
    STEP("move QPs to INIT");
    qp_to_init(qp0, port);
    qp_to_init(qp1, port);

    // gather addressing info (LID + GID)
    uint16_t dlid = port_attr.lid;
    uint32_t psn0 = rand32() & 0xffffff;
    uint32_t psn1 = rand32() & 0xffffff;

    STEP("connect QPs (RTR/RTS)");
    // connect both ways (qp0 -> qp1, qp1 -> qp0)
    qp_to_rtr(qp0, qp1->qp_num, dlid, gid, port, IBV_MTU_1024, psn1);
    qp_to_rtr(qp1, qp0->qp_num, dlid, gid, port, IBV_MTU_1024, psn0);
    qp_to_rts(qp0, psn0);
    qp_to_rts(qp1, psn1);

    /*
        RDMA verbs — three levers you use all the time

        • ibv_post_recv(qp, wrs, ...)  — put Receive WQEs on a QP’s receive queue
        • ibv_post_send(qp, wrs, ...)  — put Send / RDMA / Atomic WQEs on a QP’s send queue
        • ibv_poll_cq(cq, n, wc_array) — pull completions (CQEs) off a completion queue

        Below is a tight mental model that maps these APIs to the “modes”
        (two-sided send/recv vs one-sided read/write).

        ⸻

        0) the minimum objects you always have
        • MR: memory region you registered → gives you lkey (local) and, if you export it, rkey (remote).
        • QP: queue pair → has a send and recv queue.
        • CQ: completion queue → where finished WRs produce CQEs.
        • SGE: scatter/gather entry → points at your payload buffer (addr, length, lkey).

        You post work requests (WRs) that reference one or more SGEs. You get work completions (WC/CQE)
        back with status, byte_len, opcode, wr_id, etc.

        ⸻

        1) two-sided messaging (SEND/RECV)

        Pattern: receiver posts RECVs ahead of time; sender posts SENDs; both sides see completions.

        Receiver (must be posted before SEND arrives):
            struct ibv_sge     r_sge = { .addr=(uintptr_t)rx_buf, .length=len, .lkey=rx_mr->lkey };
            struct ibv_recv_wr r_wr  = { .wr_id=RID, .sg_list=&r_sge, .num_sge=1 };
            struct ibv_recv_wr *bad;
            ibv_post_recv(qp, &r_wr, &bad);          // enqueue a receive WQE

        Sender:
            struct ibv_sge     s_sge = { .addr=(uintptr_t)tx_buf, .length=len, .lkey=tx_mr->lkey };
            struct ibv_send_wr s_wr  = {0};
            s_wr.wr_id      = SID;
            s_wr.sg_list    = &s_sge;
            s_wr.num_sge    = 1;
            s_wr.opcode     = IBV_WR_SEND;           // two-sided send
            s_wr.send_flags = IBV_SEND_SIGNALED;     // ask for a CQE on sender
            // optionally: IBV_SEND_INLINE for tiny messages
            struct ibv_send_wr *bad2;
            ibv_post_send(qp, &s_wr, &bad2);

        Polling for completions (both sides):
            struct ibv_wc wc;
            int n = ibv_poll_cq(cq, 1, &wc);         // returns 0/1
            // wc.status == IBV_WC_SUCCESS, wc.opcode tells you SEND/RECV/…
            // Receive side sees a WC with opcode == IBV_WC_RECV.

        Key points:
        • RECVs consume a posted WQE each time a SEND arrives (so keep a backlog).
        • Both sides can get CQEs: SENDer (if SIGNALED), RECEIVer (always, when a RECV completes).
        • This is the “both participate” style — data is copied into the posted RECV buffer.

        ⸻

        2) one-sided RDMA WRITE (initiator pushes data into remote memory)

        Pattern: only the initiator posts a send-queue WR of type RDMA_WRITE. The remote does not post a RECV.
        The remote must have previously registered the destination MR with IBV_ACCESS_REMOTE_WRITE and given
        you (remote_addr, rkey).

        Initiator:
            struct ibv_sge     s_sge = { .addr=(uintptr_t)local_src, .length=len, .lkey=local_mr->lkey };
            struct ibv_send_wr s_wr  = {0};
            s_wr.wr_id   = SID;
            s_wr.sg_list = &s_sge;
            s_wr.num_sge = 1;
            s_wr.opcode  = IBV_WR_RDMA_WRITE;
            s_wr.wr.rdma.remote_addr = remote_addr;  // peer's VA
            s_wr.wr.rdma.rkey        = remote_rkey;  // peer's MR rkey
            s_wr.send_flags = IBV_SEND_SIGNALED;
            ibv_post_send(qp, &s_wr, &bad2);

        Completions:
        • Only the initiator sees a CQE (if signaled).
        • The remote side does not get a RECV completion; its memory just changes.

        Use cases: put data directly into the peer’s buffers (e.g., ring buffers, ready flags, tensor payloads)
        with zero remote CPU involvement.

        ⸻

        3) one-sided RDMA READ (initiator pulls data from remote memory)

        Pattern: only the initiator posts RDMA_READ. Remote still does not post a RECV.
        Remote MR needs IBV_ACCESS_REMOTE_READ.

            struct ibv_sge     s_sge = { .addr=(uintptr_t)local_dst, .length=len, .lkey=local_mr->lkey };
            struct ibv_send_wr s_wr  = {0};
            s_wr.wr_id   = SID;
            s_wr.sg_list = &s_sge;
            s_wr.num_sge = 1;
            s_wr.opcode  = IBV_WR_RDMA_READ;
            s_wr.wr.rdma.remote_addr = remote_addr;
            s_wr.wr.rdma.rkey        = remote_rkey;
            s_wr.send_flags = IBV_SEND_SIGNALED;
            ibv_post_send(qp, &s_wr, &bad2);

        Completions:
        • Only the initiator gets the CQE; remote doesn’t see it.

        Use cases: pull state/headers or read remote GPU/host buffers as needed.

        ⸻

        4) immediate data & “notify without copying”

        There’s also SEND with immediate:
        • Sender uses IBV_WR_SEND_WITH_IMM and sets wr.imm_data.
        • Receiver must have a posted RECV; it gets a RECV CQE with wc.wc_flags & IBV_WC_WITH_IMM and the 32-bit imm_data.
        This is handy to deliver small control info while still using two-sided delivery.

        ⸻

        6) the three APIs in practice

        ibv_post_recv(qp, wr, &bad)
        • Use only for two-sided messaging.
        • Pre-post a pool of RECV buffers.
        • Each incoming SEND will consume exactly one RECV WQE.

        ibv_post_send(qp, wr, &bad)
        • Use for everything on the send side: IBV_WR_SEND, IBV_WR_SEND_WITH_IMM, IBV_WR_RDMA_WRITE, IBV_WR_RDMA_READ, IBV_WR_ATOMIC_*.
        • Mark small SENDs as IBV_SEND_INLINE if the device supports it (cap.max_inline_data).
        • Signal only some WRs (e.g., every Nth) to reduce CQ pressure.

        ibv_poll_cq(cq, n, wc)
        • Returns completed WRs (from either the send or receive side of any QP linked to that CQ).
        • Check wc.status (== IBV_WC_SUCCESS), wc.opcode (e.g., IBV_WC_RECV, IBV_WC_SEND, IBV_WC_RDMA_WRITE, IBV_WC_RDMA_READ),
          wc.wr_id (your cookie), and wc.byte_len (recv path).

        (You can also use event-driven completions via a completion channel, but polling is the simplest to start.)
    */
    STEP("post RECV on qp1");
    // Post one RECV on qp1
    struct ibv_sge r_sge = {.addr=(uintptr_t)buf1, .length=(uint32_t)SZ, .lkey=mr1->lkey};
    struct ibv_recv_wr r_wr = {.wr_id=0xBEEF, .sg_list=&r_sge, .num_sge=1};
    struct ibv_recv_wr *bad_rwr = NULL;
    CEQ(ibv_post_recv(qp1, &r_wr, &bad_rwr), "post_recv failed");

    STEP("post SEND on qp0");
    // Post one SEND on qp0
    struct ibv_sge s_sge = {.addr=(uintptr_t)buf0, .length=(uint32_t)SZ, .lkey=mr0->lkey};
    struct ibv_send_wr s_wr = {0};
    s_wr.wr_id = 0xCAFE;
    s_wr.sg_list = &s_sge;
    s_wr.num_sge = 1;
    s_wr.opcode = IBV_WR_SEND;
    s_wr.send_flags = IBV_SEND_SIGNALED;
    struct ibv_send_wr *bad_swr = NULL;
    CEQ(ibv_post_send(qp0, &s_wr, &bad_swr), "post_send failed");

    /*
        🕑 Blocking or not?

        ibv_poll_cq() is non-blocking.
        • If there’s at least one CQE available, it immediately returns it to you.
        • If there’s nothing in the CQ, it immediately returns 0 (no completions).
        • If something goes wrong, it returns <0.

        If you want blocking behavior (i.e., wait until a CQE arrives), you use event-driven mode with a completion channel.
    */
    STEP("poll CQ for both completions");
    // poll completions for both
    int got = 0;
    while (got < 2) {
        struct ibv_wc wc;
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n > 0) {
            CEQ(wc.status != IBV_WC_SUCCESS, "WC bad status");
            got++;
            printf("CQE: wr_id=0x%lx, opcode=%d (got %d/2)\n", wc.wr_id, wc.opcode, got);
        }
    }

    STEP("done");
    printf("buf1 now contains: '%s'\n", (char*)buf1);
    return 0;
}