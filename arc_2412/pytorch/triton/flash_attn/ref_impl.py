# Truncated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, # [64, 1024, 8, 16]
    K, # [64, 1024, 8, 16]
    V, # [64, 1024, 8, 16]
    Out, # [64, 1024, 8, 16]
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb, # 1024 * 8 * 16
    stride_qh, # 16
    stride_qm, # 8 * 16 = 128
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads, # 8
    seqlen_q, # 1024
    seqlen_k, # 1024
    headdim, # 16
    IS_CAUSAL: tl.constexpr, 
    BLOCK_HEADDIM: tl.constexpr, # 16
    BLOCK_M: tl.constexpr, # 128
    BLOCK_N: tl.constexpr, # 128
):
    # grid [seqlen_q / BLOCK_M, batch * nheads] -> [1024/128, 64*8] = [8, 512]
    start_m = tl.program_id(0) # 3 (from 0~7) block id
    off_hb = tl.program_id(1) # 21 (from 0~511)
    off_b = off_hb // nheads # offset_batch = 21 // 8 = 2
    off_h = off_hb % nheads # offset_head = 21 % 8 = 5
    # initialize offsets
    # **********************************************
    # Q's seqlen is divided into BLOCKs. corresponding offsets start from start_m
    # K/V starts from the beginning of the sequence
    # **********************************************
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # 3 * 128 + list(0, 128) = [384, 385, ..., 511]
    offs_n = tl.arange(0, BLOCK_N) # [0, 1, ..., 127]
    offs_d = tl.arange(0, BLOCK_HEADDIM) # [0, 1, ..., 15]
    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    # *** broadcasted index plus strided offsets ***
    # Q + 2 * (1024 * 8 * 16) + 5 * 16 + [(384, 385, ..., 511) * 128, None] + [None, (0, 1, ..., 15)]
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N): # for start_n in range(0, 1024, 128), e.g. 384
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True) # [BLOCK, BLOCK]
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            # (384, 385, ..., 511)[:, None] >= (384, 385, ..., 511)[None, :] 
            # -->
            # [[T, F, F, ..., F], 
            #  [T, T, F, ..., F],
            #  [T, T, T, ..., F],
            #  ...
            #  [T, T, T, ..., T]]
            # -->
            # [[0, -inf, -inf, ..., -inf],
            #  [0, 0, -inf, ..., -inf],
            #  [0, 0, 0, ..., -inf],
            #  ...
            #  [0, 0, 0, ..., 0]]
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i) # [BLOCK]
        p = tl.exp(qk * softmax_scale - m_ij[:, None]) # exp(qk - m_ij) # [BLOCK, BLOCK]
        l_ij = tl.sum(p, 1) # [BLOCK]

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        v = tl.load(v_ptrs + start_n * stride_vn) # [BLOCK, d_head]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v) # [BLOCK, d_head]

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    tl.store(out_ptrs, acc_o)

    
def _flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape # [64, 1024, 8, 32]
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    assert seqlen_q % 128 == 0, "seqlen_q should be multiple of 128"
    lse = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert d == BLOCK_HEADDIM, "d should be equal to BLOCK_HEADDIM"
    
    BLOCK = 128
    assert seqlen_q % BLOCK == 0, "seqlen_q should be multiple of 128"
    assert seqlen_k % BLOCK == 0, "seqlen_k should be multiple of 128"
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q, # [64, 1024, 8, 32]
        k, # [64, 1024, 8, 32]
        v, # [64, 1024, 8, 32]
        o, # [64, 1024, 8, 32]
        lse,
        tmp,
        softmax_scale,
        q.stride(0), # 1024 * 8 * 32 = 262144
        q.stride(2), # 32
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        d,
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o


flash_attn_func = FlashAttnFunc.apply
