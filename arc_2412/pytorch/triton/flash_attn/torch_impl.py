import torch


def torch_native(q, k, v):
    batch, q_len, n_heads, d_head = q.shape
    _, kv_len, _, _ = k.shape
    q = q.permute(0, 2, 1, 3).reshape(batch * n_heads, q_len, d_head)
    k = k.permute(0, 2, 1, 3).reshape(batch * n_heads, kv_len, d_head)
    v = v.permute(0, 2, 1, 3).reshape(batch * n_heads, kv_len, d_head)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    o = o.view(batch, n_heads, q_len, d_head).permute(0, 2, 1, 3)
    return o


def torch_manual(q, k, v):
    batch, q_len, n_heads, d_head = q.shape
    _, kv_len, _, _ = k.shape
    q = q.permute(0, 2, 1, 3).reshape(batch * n_heads, q_len, d_head)
    k = k.permute(0, 2, 1, 3).reshape(batch * n_heads, kv_len, d_head)
    v = v.permute(0, 2, 1, 3).reshape(batch * n_heads, kv_len, d_head)
    s = q @ k.transpose(1, 2) / (d_head**0.5)
    p = torch.nn.functional.softmax(s, dim=-1)
    o = p @ v
    o = o.view(batch, n_heads, q_len, d_head).permute(0, 2, 1, 3)
    return o


def _fwd_kernel(q, k, v, o, tid_0, tid_1, tid_2, Q_BLOCK, KV_BLOCK):
    batch, q_len, n_heads, d_head = q.shape
    _, kv_len, _, _ = k.shape
    batch_idx = tid_0
    head_idx = tid_1
    q_block_idx = tid_2

    q_block = q[
        batch_idx, q_block_idx * Q_BLOCK : (q_block_idx + 1) * Q_BLOCK, head_idx
    ]  # [Q_BLOCK, d_head]
    acc = None
    m = torch.full((Q_BLOCK,), float("-inf"))  # [Q_BLOCK]
    l = torch.full((Q_BLOCK,), float("-inf"))  # [Q_BLOCK]
    n_kv_blocks = kv_len // KV_BLOCK
    for j in range(n_kv_blocks):
        k_block = k[batch_idx, j * KV_BLOCK : (j + 1) * KV_BLOCK, head_idx]  # [KV_BLOCK, d_head]
        v_block = v[batch_idx, j * KV_BLOCK : (j + 1) * KV_BLOCK, head_idx]  # [KV_BLOCK, d_head]
        s = q_block @ k_block.T / (d_head**0.5)  # [Q_BLOCK, KV_BLOCK]
        m_j = torch.max(s.max(dim=-1).values, m)  # [Q_BLOCK]
        if j == 0:
            acc = torch.exp(s - m_j[:, None]) @ v_block  # [Q_BLOCK, d_head]
            l = torch.exp(s - m_j[:, None]).sum(dim=-1)  # [Q_BLOCK]
        else:
            acc *= torch.exp(m - m_j)[:, None]  # [Q_BLOCK, d_head]
            acc += torch.exp(s - m_j[:, None]) @ v_block  # [Q_BLOCK, d_head]
            l *= torch.exp(m - m_j)  # [Q_BLOCK]
            l += torch.exp(s - m_j[:, None]).sum(dim=-1)  # [Q_BLOCK]
        m = m_j
    acc /= l[:, None]  # [Q_BLOCK, d_head]
    o[batch_idx, q_block_idx * Q_BLOCK : (q_block_idx + 1) * Q_BLOCK, head_idx] = acc
    return


def flash_attn_torch(q, k, v):
    batch, q_len, n_heads, d_head = q.shape
    o = torch.zeros_like(q)
    Q_BLOCK = 32
    KV_BLOCK = 32
    for tid_0 in range(batch):
        for tid_1 in range(n_heads):
            for tid_2 in range(q_len // Q_BLOCK):
                _fwd_kernel(q, k, v, o, tid_0, tid_1, tid_2, Q_BLOCK, KV_BLOCK)
    return o


def main():
    torch.manual_seed(7)
    batch_size = 4
    q_len = 1024
    kv_len = 1024
    n_heads = 8
    d_head = 32
    q = torch.randn(batch_size, q_len, n_heads, d_head)
    k = torch.randn(batch_size, kv_len, n_heads, d_head)
    v = torch.randn(batch_size, kv_len, n_heads, d_head)

    ref1 = torch_native(q, k, v)
    ref2 = torch_manual(q, k, v)
    out = flash_attn_torch(q, k, v)
    diff = torch.abs(ref1 - ref2)
    rel = diff / torch.abs(ref2)
    print(f"ref1 <> ref2: {diff.max()}, {rel.max()}")
    diff = torch.abs(ref1 - out)
    rel = diff / torch.abs(out)
    print(f"ref1 <> out: {diff.max()}, {rel.max()}")
    assert torch.allclose(ref1, ref2, atol=1e-5)
    assert torch.allclose(ref1, out, atol=1e-5)


if __name__ == "__main__":
    main()
