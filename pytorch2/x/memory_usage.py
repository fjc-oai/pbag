import fire
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

enable_activation_checkpointing = False


def get_ac():
    global enable_activation_checkpointing
    return enable_activation_checkpointing


def set_ac(val):
    global enable_activation_checkpointing
    enable_activation_checkpointing = val


# Multi-Head Attention Module
class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0, "Embedding size must be divisible by the number of heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        Q = self.q_proj(x)  # 8*4*4096*32*4=16MB
        K = self.k_proj(x)  # 8*4*4096*32*4=16MB
        V = self.v_proj(x)  # 8*4*4096*32*4=16MB

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)  # 8*4*4096*4096*4=2G
        # attention = torch.softmax(scores, dim=-1)  # 8*4*4096*4096*4=2GB
        # out = (
        #     (attention @ V).transpose(1, 2).reshape(batch_size, seq_len, d_model)
        # )  # 8*4096*128*4=16MB
        # return self.out_proj(out)
        x = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)  # 8*4*4096*4096*4=2G
        x = torch.softmax(x, dim=-1)  # 8*4*4096*4096*4=2GB
        x = (x @ V).transpose(1, 2).reshape(batch_size, seq_len, d_model)  # 8*4096*128*4=16MB
        return self.out_proj(x)


# Feedforward MLP Module
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # 8*4096*128*4=16MB
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Transformer Encoder Layer
class AttnMlpBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttnMlpBlock, self).__init__()
        self.attention = Attention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_out = self.attention(x)  # 8*4096*128*4=16MB
        x = self.norm1(x + attention_out)  # 8*4096*128*4=16MB

        ff_out = self.feed_forward(x)  # 8*4096*128*4=16MB
        x = self.norm2(x + ff_out)  # 8*4096*128*4=16MB
        return x


# Complete LLM Model
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([AttnMlpBlock(d_model, n_heads) for _ in range(n_layers)])
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)  # 8*4096*128*4=16MB
        for layer in self.layers:
            enable_activation_checkpointing = get_ac()
            if enable_activation_checkpointing:
                x = checkpoint.checkpoint(layer, x)  # 8*4096*128*4=16MB
            else:
                x = layer(x)  # 8*4096*128*4=16MB
        x = checkpoint.checkpoint(self.unembed, x)  # 8*4096*10240*4=1280MB
        # x = self.unembed(x)  # 8*4096*10240*4=1280MB
        return x


vocab_size = 10240
d_model = 128
n_heads = 4
n_layers = 4
n_ctx = 4096
batch_size = 8
mb_size = 8


def fwd_only():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model = Transformer(vocab_size, d_model, n_heads, n_layers).cuda()
        x = torch.randint(0, vocab_size, (batch_size, n_ctx)).cuda()  # 8*4096*4=128KB

        torch.cuda.memory._record_memory_history()
        output = model(x)

        torch.cuda.memory._dump_snapshot(f"fwd_only_snapshot.pickle")
        print(f"Saving forward pass snapshot to fwd_only_snapshot.pickle")
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage during forward pass: {peak_memory / 1024**3:.2f} GB")


def fwd_bwd():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Transformer(vocab_size, d_model, n_heads, n_layers).cuda()
    x = torch.randint(0, vocab_size, (batch_size, n_ctx)).cuda()  # 8*4096*4=128KB

    torch.cuda.memory._record_memory_history()
    output = model(x)
    loss = output.sum()
    loss.backward()
    torch.cuda.memory._dump_snapshot(f"fwd_bwd_snapshot.pickle")
    print(f"Saving forward-backward pass snapshot to fwd_bwd_snapshot.pickle")
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage during forward-backward pass: {peak_memory / 1024**3:.2f} GB")


def fwd_bwd_activation_checkpoint():
    set_ac(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Transformer(vocab_size, d_model, n_heads, n_layers).cuda()
    x = torch.randint(0, vocab_size, (batch_size, n_ctx)).cuda()  # 8*4096*4=128KB

    torch.cuda.memory._record_memory_history()
    output = model(x)
    loss = output.sum()
    del output
    loss.backward()
    torch.cuda.memory._dump_snapshot(f"fwd_bwd_ac_snapshot.pickle")
    print(f"Saving forward-backward pass snapshot to fwd_bwd_ac_snapshot.pickle")
    peak_memory = torch.cuda.max_memory_allocated()
    print(
        f"Peak memory usage during forward-backward pass with activation checkpointing: {peak_memory / 1024**3:.2f} GB"
    )


def profile(mode):
    fns = {
        "fwd_only": fwd_only,
        "fwd_bwd": fwd_bwd,
        "fwd_bwd_ac": fwd_bwd_activation_checkpoint,
    }
    fns[mode]()


if __name__ == "__main__":
    fire.Fire(profile)
