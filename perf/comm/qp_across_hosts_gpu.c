// qp_across_hosts_gpu.c
// Minimal two-node RC demo using GPUDirect RDMA:
// - Same flow as qp_across_hosts.c but buffers live in GPU memory
// - Creates PD/CQ/QP
// - Registers one GPU buffer on each side (requires GPUDirect RDMA support)
// - Exchanges {QPN, LID, GID, PSN, RKey, VAddr} via tiny TCP
// - Moves QP INIT->RTR->RTS
// - CLIENT issues an RDMA WRITE from its GPU buffer into SERVER GPU buffer
// - SERVER copies GPU buffer back to host and prints what landed
//
// Build (examples; adjust CUDA paths as needed):
//   nvcc -O2 qp_across_hosts_gpu.c -Xcompiler -Wall -DUSE_CUDA_DRIVER -libverbs -lcudart -lcuda -o qp_across_hosts_gpu
//
// Run:    server: ./qp_across_hosts_gpu server 18515
//         client: ./qp_across_hosts_gpu client <server-ip> 18515
//
// Notes:
// - Requires GPUDirect RDMA capable NIC + driver + nvidia-peermem (or equivalent) kernel module.
// - If ibv_reg_mr on a CUDA pointer fails, your environment likely lacks GPUDirect RDMA support.

#define _POSIX_C_SOURCE 200112L
#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <cuda_runtime_api.h>
#ifdef USE_CUDA_DRIVER
#include <cuda.h>
#endif

#define CHECK(cnd, msg) do { if (cnd) { perror(msg); exit(1);} } while (0)
#define DIE(msg) do { fprintf(stderr, "ERR: %s\n", msg); exit(1);} while (0)
#define CUDA_CHECK(cmd) do { \
  cudaError_t _e = (cmd); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERR: %s failed: %s (%d)\n", #cmd, cudaGetErrorString(_e), (int)_e); \
    exit(1); \
  } \
} while (0)
#ifdef USE_CUDA_DRIVER
#define CU_CHECK(cmd) do { \
  CUresult _r = (cmd); \
  if (_r != CUDA_SUCCESS) { \
    const char* _s = NULL; \
    cuGetErrorString(_r, &_s); \
    fprintf(stderr, "CUDA-DRIVER ERR: %s failed: %s (%d)\n", #cmd, _s?_s:"?", (int)_r); \
    exit(1); \
  } \
} while (0)
#endif

static const uint8_t  PORT_NUM  = 1;   // HCA port
static const uint32_t MSG_SIZE  = 4096;
static const int      GID_INDEX = 0;   // pick a valid GID for RoCE; OK for IB too

static uint32_t rand32_psn() {
  uint32_t x = (uint32_t)rand();
  return x & 0xFFFFFF; // 24-bit PSN
}

static int tcp_listen(uint16_t port) {
  int s = socket(AF_INET, SOCK_STREAM, 0); CHECK(s<0, "socket");
  int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  CHECK(bind(s, (struct sockaddr*)&addr, sizeof(addr))<0, "bind");
  CHECK(listen(s, 1)<0, "listen");
  return s;
}
static int tcp_accept(int ls) {
  struct sockaddr_in peer; socklen_t len = sizeof(peer);
  int cs = accept(ls, (struct sockaddr*)&peer, &len); CHECK(cs<0, "accept");
  return cs;
}
static int tcp_connect(const char* ip, uint16_t port) {
  int s = socket(AF_INET, SOCK_STREAM, 0); CHECK(s<0,"socket");
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  CHECK(inet_pton(AF_INET, ip, &addr.sin_addr)<=0, "inet_pton");
  CHECK(connect(s, (struct sockaddr*)&addr, sizeof(addr))<0, "connect");
  return s;
}
static void xfer_all(int fd, void* buf, size_t len, int send_flag) {
  uint8_t* p = (uint8_t*)buf; size_t off=0;
  while (off < len) {
    ssize_t n = send_flag ? send(fd, p+off, len-off, 0) : recv(fd, p+off, len-off, MSG_WAITALL);
    CHECK(n<=0, send_flag? "send":"recv");
    off += (size_t)n;
  }
}

// host/network helpers for u64
static uint64_t htonll_u64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return ((uint64_t)htonl((uint32_t)(x>>32))) | (((uint64_t)htonl((uint32_t)x))<<32);
#else
  return x;
#endif
}
static uint64_t ntohll_u64(uint64_t x) { return htonll_u64(x); }

// What we exchange over TCP
struct __attribute__((packed)) ConnInfo {
  uint32_t qpn;      // qp number
  uint16_t lid;      // InfiniBand LID (0 for RoCE)
  uint8_t  gid[16];  // GID (RoCE or IB GRH)
  uint32_t psn;      // starting PSN
  uint32_t rkey;     // MR rkey
  uint64_t vaddr;    // MR base virtual address (GPU VA here)
};

// (fill_gid removed; using memcpy directly)
static void to_wire(struct ConnInfo* w) {
  w->qpn  = htonl(w->qpn);
  w->lid  = htons(w->lid);
  w->psn  = htonl(w->psn);
  w->rkey = htonl(w->rkey);
  w->vaddr= htonll_u64(w->vaddr);
}
static void from_wire(struct ConnInfo* w) {
  w->qpn  = ntohl(w->qpn);
  w->lid  = ntohs(w->lid);
  w->psn  = ntohl(w->psn);
  w->rkey = ntohl(w->rkey);
  w->vaddr= ntohll_u64(w->vaddr);
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
  CHECK(r, "modify_qp INIT");
}
static void qp_to_rtr(struct ibv_qp* qp,
                      const struct ConnInfo* peer,
                      uint8_t port, enum ibv_mtu mtu) {
  struct ibv_qp_attr a = {0};
  a.qp_state           = IBV_QPS_RTR;
  a.path_mtu           = mtu;
  a.dest_qp_num        = peer->qpn;
  a.rq_psn             = peer->psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer      = 12;

  a.ah_attr.port_num   = port;
  a.ah_attr.sl         = 0;
  a.ah_attr.is_global  = 1; // use GRH (works for RoCE and IB GRH)
  memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
  a.ah_attr.grh.sgid_index = GID_INDEX;
  a.ah_attr.grh.hop_limit  = 1;
  a.ah_attr.dlid       = peer->lid; // for RoCE this is ignored

  int r = ibv_modify_qp(qp, &a,
    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  CHECK(r, "modify_qp RTR");
}
static void qp_to_rts(struct ibv_qp* qp, uint32_t sq_psn) {
  struct ibv_qp_attr a = {0};
  a.qp_state      = IBV_QPS_RTS;
  a.timeout       = 14;
  a.retry_cnt     = 7;
  a.rnr_retry     = 7;
  a.sq_psn        = sq_psn;
  a.max_rd_atomic = 1;
  int r = ibv_modify_qp(qp, &a,
    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  CHECK(r, "modify_qp RTS");
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage:\n  server: %s server <port>\n  client: %s client <server-ip> <port>\n", argv[0], argv[0]);
    return 2;
  }
  srand((unsigned)time(NULL));

  int is_server = !strcmp(argv[1], "server");
  char ip[64] = {0};
  uint16_t port = 0;
  if (is_server) {
    port = (uint16_t)atoi(argv[2]);
  } else {
    if (argc < 4) DIE("client mode needs <server-ip> <port>");
    strncpy(ip, argv[2], sizeof(ip)-1);
    port = (uint16_t)atoi(argv[3]);
  }

  // 0) CUDA setup (ensure driver + runtime contexts are initialized)
#ifdef USE_CUDA_DRIVER
  CU_CHECK(cuInit(0));
#endif
  CUDA_CHECK(cudaSetDevice(0));
  // Force-create context to avoid lazy init races with peer-memory hooks
  CUDA_CHECK(cudaFree(0));
  int cur_dev = -1; CUDA_CHECK(cudaGetDevice(&cur_dev));
  struct cudaDeviceProp cprop; CUDA_CHECK(cudaGetDeviceProperties(&cprop, cur_dev));
  int uva = 0; CUDA_CHECK(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, cur_dev));
  fprintf(stderr, "[cuda] device %d: '%s' UVA=%d CC=%d.%d\n", cur_dev, cprop.name, uva, cprop.major, cprop.minor);

  // 1) Open device/context
  int num_dev = 0;
  struct ibv_device **devs = ibv_get_device_list(&num_dev);
  CHECK(!devs || num_dev==0, "ibv_get_device_list");
  int chosen_idx = -1;
  for (uint32_t i = 0; i < num_dev; i++) {
    const char *name = ibv_get_device_name(devs[i]);
    if (name && strncmp(name, "mlx5_ib", 7) == 0) {
      chosen_idx = i;
      break;
    }
  }
  CHECK(chosen_idx < 0, "no usable RDMA device");
  struct ibv_context *ctx = ibv_open_device(devs[chosen_idx]);
  CHECK(!ctx, "ibv_open_device");

  // 2) Query port + GID
  struct ibv_port_attr pattr;
  CHECK(ibv_query_port(ctx, PORT_NUM, &pattr), "ibv_query_port");
  union ibv_gid gid; memset(&gid, 0, sizeof(gid));
  CHECK(ibv_query_gid(ctx, PORT_NUM, GID_INDEX, &gid), "ibv_query_gid");

  // 3) PD, CQ, MR (on GPU memory)
  struct ibv_pd *pd = ibv_alloc_pd(ctx);              CHECK(!pd, "alloc_pd");
  struct ibv_cq *cq = ibv_create_cq(ctx, 32, NULL, NULL, 0);
  CHECK(!cq, "create_cq");

  void *gpu_buf = NULL;
  void *host_buf = NULL; // fallback staging when GPU MR not available
  int use_gpu_mr = 1;    // whether our locally registered MR points to GPU memory
  CUDA_CHECK(cudaMalloc(&gpu_buf, MSG_SIZE));
  CUDA_CHECK(cudaMemset(gpu_buf, 0, MSG_SIZE));
  // Sanity check pointer attributes (helpful diagnostics for GPUDirect)
  struct cudaPointerAttributes cptr_attr;
  CUDA_CHECK(cudaPointerGetAttributes(&cptr_attr, gpu_buf));
#if CUDART_VERSION >= 10000
  int is_device_ptr = (cptr_attr.type == cudaMemoryTypeDevice);
#else
  int is_device_ptr = (cptr_attr.memoryType == cudaMemoryTypeDevice);
#endif
  fprintf(stderr, "[cuda] gpu_buf=%p is_device_ptr=%d\n", gpu_buf, is_device_ptr);
  const char *MSG = "hello-from-client-via-RDMA-WRITE-GPU";

  // Register GPU MR; allow remote read/write for demonstration
  int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  struct ibv_mr *mr = ibv_reg_mr(pd, gpu_buf, MSG_SIZE, mr_flags);
  if (!mr) {
    int saved_errno = errno;
    fprintf(stderr, "[warn] ibv_reg_mr(GPU) failed: %s (%d)\n", strerror(saved_errno), saved_errno);
    // Optional strict mode to fail-fast when GPUDirect RDMA is expected
    const char* strict = getenv("QP_GPU_STRICT");
    if (strict && atoi(strict) == 1) {
      fprintf(stderr, "[error] QP_GPU_STRICT=1 set; not falling back. Exiting.\n");
      exit(1);
    }
    // Fallback: use host pinned memory as our MR and do explicit D2H/H2D copies
    use_gpu_mr = 0;
    fprintf(stderr, "[info] Falling back to host-pinned MR (cudaMallocHost).\n");
    CUDA_CHECK(cudaMallocHost(&host_buf, MSG_SIZE));
    memset(host_buf, 0, MSG_SIZE);
    mr = ibv_reg_mr(pd, host_buf, MSG_SIZE, mr_flags);
    CHECK(!mr, "ibv_reg_mr host-pinned");
  }

  // 4) QP
  struct ibv_qp_init_attr qia = {0};
  qia.qp_type = IBV_QPT_RC;
  qia.send_cq = cq; qia.recv_cq = cq;
  qia.cap.max_send_wr = 32; qia.cap.max_recv_wr = 32;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  qia.cap.max_inline_data = 220;

  struct ibv_qp *qp = ibv_create_qp(pd, &qia);
  CHECK(!qp, "create_qp");

  // 5) INIT
  qp_to_init(qp, PORT_NUM);

  // 6) exchange connection info over TCP
  struct ConnInfo me = {0}, peer = {0};
  me.qpn  = qp->qp_num;
  me.lid  = pattr.lid;            // 0 on RoCE; IB uses it
  me.psn  = rand32_psn();
  me.rkey = mr->rkey;
  me.vaddr= (uintptr_t)(use_gpu_mr ? gpu_buf : host_buf);   // local MR address (GPU or host)
  memcpy(me.gid, gid.raw, 16);

  int sock = -1, ls = -1;
  if (is_server) {
    ls = tcp_listen(port);
    fprintf(stderr, "[server] listening on %u\n", port);
    sock = tcp_accept(ls);
    fprintf(stderr, "[server] accepted connection\n");

    struct ConnInfo tmp;
    // server sends first (arbitrary), then receives peer
    struct ConnInfo sendme = me; to_wire(&sendme);
    xfer_all(sock, &sendme, sizeof(sendme), 1);
    xfer_all(sock, &tmp, sizeof(tmp), 0);
    peer = tmp; from_wire(&peer);
  } else {
    sock = tcp_connect(ip, port);
    fprintf(stderr, "[client] connected to %s:%u\n", ip, port);

    struct ConnInfo tmp;
    // client receives first (arbitrary), then sends
    xfer_all(sock, &tmp, sizeof(tmp), 0);
    peer = tmp; from_wire(&peer);

    struct ConnInfo sendme = me; to_wire(&sendme);
    xfer_all(sock, &sendme, sizeof(sendme), 1);
  }

  fprintf(stderr, "[%s] local  qpn=0x%x lid=0x%x psn=0x%x rkey=0x%x vaddr=0x%lx (%s)\n",
    is_server? "server":"client", me.qpn, me.lid, me.psn, me.rkey, (unsigned long)me.vaddr,
    use_gpu_mr? "GPU":"HOST-PINNED");
  fprintf(stderr, "[%s] remote qpn=0x%x lid=0x%x psn=0x%x rkey=0x%x vaddr=0x%lx (GPU)\n",
    is_server? "server":"client", peer.qpn, peer.lid, peer.psn, peer.rkey, (unsigned long)peer.vaddr);

  // 7) RTR/RTS
  qp_to_rtr(qp, &peer, PORT_NUM, IBV_MTU_1024);
  qp_to_rts(qp, me.psn);

  // 8) Demo operation using GPU memory
  if (is_server) {
    // SERVER: wait a moment; client will RDMA_WRITE into our buffer (GPU or host).
    sleep(1);
    if (use_gpu_mr) {
      char tmp_host[MSG_SIZE];
      CUDA_CHECK(cudaMemcpy(tmp_host, gpu_buf, MSG_SIZE, cudaMemcpyDeviceToHost));
      tmp_host[MSG_SIZE-1] = '\0';
      fprintf(stderr, "[server] buffer after write (GPU->host copy): '%s'\n", tmp_host);
    } else {
      ((char*)host_buf)[MSG_SIZE-1] = '\0';
      fprintf(stderr, "[server] buffer after write (HOST-PINNED): '%s'\n", (char*)host_buf);
    }
  } else {
    // CLIENT: prepare source buffer then RDMA_WRITE from our MR (GPU or host) into server buffer
    if (use_gpu_mr) {
      CUDA_CHECK(cudaMemcpy(gpu_buf, MSG, strlen(MSG)+1, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      memcpy(host_buf, MSG, strlen(MSG)+1);
    }
    struct ibv_sge sge = {
      .addr   = (uintptr_t)(use_gpu_mr ? gpu_buf : host_buf),
      .length = MSG_SIZE,
      .lkey   = mr->lkey
    };
    struct ibv_send_wr wr = {0}, *bad = NULL;
    wr.wr_id      = 0xC11E17; // cookie
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED; // ask for CQE
    wr.wr.rdma.remote_addr = peer.vaddr; // remote address (GPU or host on peer)
    wr.wr.rdma.rkey        = peer.rkey;  // remote rkey
    CHECK(ibv_post_send(qp, &wr, &bad), "post_send RDMA_WRITE");

    // Poll for completion on client side
    struct ibv_wc wc;
    for (;;) {
      int n = ibv_poll_cq(cq, 1, &wc);
      CHECK(n<0, "poll_cq");
      if (n == 0) continue;
      if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "bad WC: status=%d opcode=%d\n", wc.status, wc.opcode);
        exit(1);
      }
      fprintf(stderr, "[client] RDMA_WRITE complete (wr_id=0x%lx)\n", wc.wr_id);
      break;
    }
  }

  // cleanup (light)
  if (sock >= 0) close(sock);
  if (ls >= 0) close(ls);
  // Optional: free GPU memory last
  if (gpu_buf) { CUDA_CHECK(cudaFree(gpu_buf)); }
  if (host_buf) { CUDA_CHECK(cudaFreeHost(host_buf)); }
  return 0;
}


