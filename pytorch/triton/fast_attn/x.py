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
    s = torch.rand(seqlen)
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
    s = torch.rand(seqlen)
    v = torch.rand(seqlen)
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
    q = torch.rand(1)
    k = torch.rand(seqlen)
    v = torch.rand(seqlen)

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
    q = torch.rand(d_head)
    k = torch.rand(seqlen, d_head)
    v = torch.rand(seqlen, d_head)

    def ref_impl(q, k, v):
        s = q @ k.T
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
        s = q @ k_block.T
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
    print("test_single_q_vector passed")


def test_full_attention():
    seqlen = 1024
    d_head = 8
    q = torch.rand(seqlen, d_head)
    k = torch.rand(seqlen, d_head)
    v = torch.rand(seqlen, d_head)

    def ref_impl(q, k, v):
        s = q @ k.T
        m = s.max(dim=1).values
        p = torch.exp(s - m)
        o = p @ v / p.sum(dim=1)[:, None]
        return o

    ref_out = ref_impl(q, k, v)

    def torch_impl(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch_out = torch_impl(q, k, v)

    BLOCK = 32
    n_blocks = seqlen // BLOCK
    out = torch.zeros_like(ref_out)
    for i in range(n_blocks):
        q_block = q[i * BLOCK : (i + 1) * BLOCK]
        acc = None
        m_i = torch.full((BLOCK,), float("-inf"))
        l = torch.full((BLOCK,), float("-inf"))
        for j in range(n_blocks):
            k_block = k[j * BLOCK : (j + 1) * BLOCK]
            v_block = v[j * BLOCK : (j + 1) * BLOCK]
            s = q_block @ k_block.T
            m_ij = torch.max(s.max(dim=1).values, m_i)
            if j == 0:
                acc = torch.exp(s - m_ij) @ v_block
                l = torch.exp(s - m_ij).sum(dim=1)
            else:
                acc *= torch.exp(m_i - m_ij)[:, None]
                acc += torch.exp(s - m_ij) @ v_block
                l *= torch.exp(m_i - m_ij)
                l += torch.exp(s - m_ij).sum(dim=1)
            m_i = m_ij
        acc /= l[:, None]
        out[i * BLOCK : (i + 1) * BLOCK] = acc
    diff = torch.abs(out - ref_out)
    rel = diff / torch.abs(ref_out)
    print(f"out <> ref_out: {diff.max()}, {rel.max()}")
    diff = torch.abs(ref_out - torch_out)
    rel = diff / torch.abs(torch_out)
    print(f"ref_out <> torch_out: {diff.max()}, {rel.max()}")

    assert torch.allclose(out, ref_out, atol=3e-2, rtol=1e-2)
    print("test_full_attention passed")


def main():
    # test_dot_attention_order()
    # test_accumuate_denominator()
    # test_accumuate_numerator()
    # test_single_q_number()
    # test_single_q_vector()
    test_full_attention()


if __name__ == "__main__":
    main()
