import torch


def test_dot_attention_order():
    n = 32
    d = 8
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    v = torch.randn(n, d)

    ref_out = q @ k.T @ v

    out = torch.zeros_like(ref_out)
    for i in range(n):
        q_i = q[i]
        out_i = torch.zeros(1, d)
        for j in range(n):
            out_i += q_i @ k[j] * v[j]
        out[i] = out_i

    assert torch.allclose(out, ref_out, atol=1e-5)
    print("test_dot_attention_order passed")


def test_accumuate_denominator():
    seqlen = 1024
    s = torch.randn(seqlen)
    m = s.max()
    ref_out = torch.exp(s - m).sum()

    BLOCK = 32
    n_blocks = seqlen // BLOCK
    acc = None
    m = torch.tensor(float("-inf"))
    for i in range(n_blocks):
        s_block = s[i * BLOCK : (i + 1) * BLOCK]
        m_i = torch.max(s_block.max(), m)
        if i == 0:
            acc = torch.exp(s_block - m_i).sum()
        else:
            acc *= torch.exp(m - m_i)
            acc += torch.exp(s_block - m_i).sum()
        m = m_i
    assert torch.allclose(acc, ref_out, atol=1e-5)
    print("test_accumuate_denominator passed")


def test_accumuate_numerator():
    seqlen = 1024
    s = torch.randn(seqlen)
    v = torch.randn(seqlen)
    m = s.max()
    rel_out = torch.exp(s - m) @ v

    BLOCK = 32
    n_blocks = seqlen // BLOCK
    acc = None
    m = torch.tensor(float("-inf"))
    for i in range(n_blocks):
        s_block = s[i * BLOCK : (i + 1) * BLOCK]
        v_block = v[i * BLOCK : (i + 1) * BLOCK]
        m_i = torch.max(s_block.max(), m)
        if i == 0:
            acc = torch.exp(s_block - m_i) @ v_block
        else:
            acc *= torch.exp(m - m_i)
            acc += torch.exp(s_block - m_i) @ v_block
        m = m_i
    assert torch.allclose(acc, rel_out, atol=1e-5)
    print("test_accumuate_numerator passed")


def test_single_q_number():
    seqlen = 1024
    q = torch.randn(1)
    k = torch.randn(seqlen)
    v = torch.randn(seqlen)

    def ref_impl(q, k, v):
        s = q * k
        m = s.max()
        p = torch.exp(s - m)
        o = p @ v / p.sum()
        return o

    ref_out = ref_impl(q, k, v)

    BLOCK = 32
    n_blocks = seqlen // BLOCK
    acc = None
    m = torch.tensor(float("-inf"))
    l = torch.tensor(float("-inf"))
    for i in range(n_blocks):
        k_block = k[i * BLOCK : (i + 1) * BLOCK]
        v_block = v[i * BLOCK : (i + 1) * BLOCK]
        s = q * k_block
        m_i = torch.max(s.max(), m)
        if i == 0:
            acc = torch.exp(s - m_i) @ v_block
            l = torch.exp(s - m_i).sum()
        else:
            acc *= torch.exp(m - m_i)
            acc += torch.exp(s - m_i) @ v_block
            l *= torch.exp(m - m_i)
            l += torch.exp(s - m_i).sum()
        m = m_i
    acc /= l
    assert torch.allclose(acc, ref_out, atol=1e-5)
    print("test_single_q_number passed")


def test_single_q_vector():
    seqlen = 1024
    d_head = 8
    q = torch.randn(d_head)
    k = torch.randn(seqlen, d_head)
    v = torch.randn(seqlen, d_head)

    def ref_impl(q, k, v):
        s = q @ k.T / (d_head ** 0.5)
        m = s.max()
        p = torch.exp(s - m)
        o = p @ v / p.sum()
        return o

    def ref_impl_torch(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q[None, :], k, v)

    ref_out = ref_impl(q, k, v)
    ref_out_torch = ref_impl_torch(q, k, v)

    BLOCK = 32
    n_blocks = seqlen // BLOCK
    acc = None
    m = torch.tensor(float("-inf"))
    l = torch.tensor(float("-inf"))
    for i in range(n_blocks):
        k_block = k[i * BLOCK : (i + 1) * BLOCK]
        v_block = v[i * BLOCK : (i + 1) * BLOCK]
        s = q @ k_block.T / (d_head ** 0.5)
        m_i = torch.max(s.max(), m)
        if i == 0:
            acc = torch.exp(s - m_i) @ v_block
            l = torch.exp(s - m_i).sum()
        else:
            acc *= torch.exp(m - m_i)
            acc += torch.exp(s - m_i) @ v_block
            l *= torch.exp(m - m_i)
            l += torch.exp(s - m_i).sum()
        m = m_i
    acc /= l

    diff = torch.abs(acc - ref_out)
    rel = diff / torch.abs(ref_out)
    print(f"acc <> ref_out: {diff.max()}, {rel.max()}")
    diff = torch.abs(ref_out - ref_out_torch)
    rel = diff / torch.abs(ref_out_torch)
    print(f"ref_out <> ref_out_torch: {diff.max()}, {rel.max()}")
    assert torch.allclose(acc, ref_out, atol=1e-5)
    print("test_single_q_vector passed")


def test_full_attention():
    q_len = 1024
    kv_len = 1024
    d_head = 8
    q = torch.randn(q_len, d_head)
    k = torch.randn(kv_len, d_head)
    v = torch.randn(kv_len, d_head)

    def ref_impl(q, k, v):
        d_head = q.shape[1]
        s = q @ k.T / (d_head ** 0.5)
        p = torch.exp(s - s.max(dim=-1).values[:, None])
        o = p @ v / p.sum(dim=1)[:, None]
        # p = torch.nn.functional.softmax(s, dim=1)
        return o

    ref_out = ref_impl(q, k, v)

    def torch_impl(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch_out = torch_impl(q, k, v)

    Q_BLOCK = 32
    KV_BLOCK = 32
    n_q_blocks = q_len // Q_BLOCK
    n_kv_blocks = kv_len // KV_BLOCK
    out = torch.zeros(q_len, d_head)
    for i in range(n_q_blocks):
        q_block = q[i * Q_BLOCK : (i + 1) * Q_BLOCK] # [Q_BLOCK, d_head]
        acc = None
        m_i = torch.full((Q_BLOCK,), float("-inf")) # [Q_BLOCK]
        l = torch.full((Q_BLOCK,), float("-inf")) # [Q_BLOCK]
        for j in range(n_kv_blocks):
            k_block = k[j * KV_BLOCK : (j + 1) * KV_BLOCK] # [KV_BLOCK, d_head]
            v_block = v[j * KV_BLOCK : (j + 1) * KV_BLOCK] # [KV_BLOCK, d_head]
            s = q_block @ k_block.T / (d_head ** 0.5) # [Q_BLOCK, KV_BLOCK]
            m_ij = torch.max(s.max(dim=-1).values, m_i) # [Q_BLOCK]
            if j == 0:
                acc = torch.exp(s - m_ij[:, None]) @ v_block # [Q_BLOCK, d_head]
                l = torch.exp(s - m_ij[:, None]).sum(dim=-1) # [Q_BLOCK]
            else:
                acc *= torch.exp(m_i - m_ij)[:, None] # [Q_BLOCK, d_head]
                acc += torch.exp(s - m_ij[:, None]) @ v_block # [Q_BLOCK, d_head]
                l *= torch.exp(m_i - m_ij) # [Q_BLOCK]
                l += torch.exp(s - m_ij[:, None]).sum(dim=-1) # [Q_BLOCK]
            m_i = m_ij # [Q_BLOCK]
        acc /= l[:, None] # [Q_BLOCK, d_head]
        out[i * Q_BLOCK : (i + 1) * Q_BLOCK] = acc
    diff = torch.abs(out - ref_out)
    rel = diff / torch.abs(ref_out)
    print(f"out <> ref_out: {diff.max()}, {rel.max()}")
    diff = torch.abs(ref_out - torch_out)
    rel = diff / torch.abs(torch_out)
    print(f"ref_out <> torch_out: {diff.max()}, {rel.max()}")

    assert torch.allclose(out, ref_out, atol=1e-5)
    print("test_full_attention passed")

