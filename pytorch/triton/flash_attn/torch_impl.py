import torch


def torch_native(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _fwd_kernel(q, k, v, o, n_heads, seqlen, d_head, tid_0, tid_1, BLOCK):
    seq_block_idx = tid_0
    off_hb = tid_1
    batch_idx = off_hb // n_heads
    head_idx = off_hb % n_heads

    q_block = q[
        batch_idx, seq_block_idx * BLOCK : (seq_block_idx + 1) * BLOCK, head_idx
    ]  # [BLOCK, d_head]
    acc = None
    m = torch.full((BLOCK,), float("-inf"))  # [BLOCK]
    l = torch.full((BLOCK,), float("-inf"))  # [BLOCK]
    n_blocks = seqlen // BLOCK
    for j in range(n_blocks):
        k_block = k[batch_idx, j * BLOCK : (j + 1) * BLOCK, head_idx]  # [BLOCK, d_head]
        v_block = v[batch_idx, j * BLOCK : (j + 1) * BLOCK, head_idx]  # [BLOCK, d_head]
        s = q_block @ k_block.T  # [BLOCK, BLOCK]
        m_j = torch.max(s.max(dim=1).values, m)  # [BLOCK]
        if j == 0:
            acc = torch.exp(s - m_j) @ v_block
            l = torch.exp(s - m_j).sum(dim=1)
        else:
            acc *= torch.exp(m - m_j)[:, None]
            acc += torch.exp(s - m_j) @ v_block
            l *= torch.exp(m - m_j)
            l += torch.exp(s - m_j).sum(dim=1)
        m = m_j
    acc /= l[:, None]
    o[batch_idx, seq_block_idx * BLOCK : (seq_block_idx + 1) * BLOCK, head_idx].copy_(acc)
    return


def flash_attn_torch(q, k, v):
    batch, seqlen, n_heads, d_head = q.shape
    assert q.shape == k.shape == v.shape
    o = torch.zeros_like(q)
    BLOCK = 32
    for tid_0 in range(seqlen // BLOCK):
        for tid_1 in range(batch * n_heads):
            print(f"Progress: {tid_0 * batch * n_heads + tid_1}/{seqlen * batch * n_heads // BLOCK}")
            _fwd_kernel(q, k, v, o, n_heads, seqlen, d_head, tid_0, tid_1, BLOCK)
    return o


def main():
    batch_size = 1
    seqlen = 128
    n_heads = 1
    d_head = 32
    q = torch.rand(batch_size, seqlen, n_heads, d_head)
    k = torch.rand(batch_size, seqlen, n_heads, d_head)
    v = torch.rand(batch_size, seqlen, n_heads, d_head)

    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = flash_attn_torch(q, k, v)
    diff = torch.abs(out - ref_out)
    rel = diff / torch.abs(ref_out)
    print(f"out <> ref_out: {diff.max()}, {rel.max()}")
    breakpoint()


if __name__ == "__main__":
    main()