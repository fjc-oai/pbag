import torch
import torch.nn as nn


# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Memory tracking for attention module
        print(f"Memory before attention projection: {torch.cuda.memory_allocated() / 1e6} MB")

        batch_size, seq_len, embed_size = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention = torch.softmax(scores, dim=-1)
        out = (attention @ V).transpose(1, 2).reshape(batch_size, seq_len, embed_size)

        print(f"Memory after attention computation: {torch.cuda.memory_allocated() / 1e6} MB")
        return self.out_proj(out)


# Feedforward MLP Module
class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        print(f"Memory before MLP: {torch.cuda.memory_allocated() / 1e6} MB")
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        print(f"Memory after MLP: {torch.cuda.memory_allocated() / 1e6} MB")
        return x


# Transformer Encoder Layer
class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        print(f"Memory before Transformer Layer: {torch.cuda.memory_allocated() / 1e6} MB")
        # Self-attention with residual connection
        attention_out = self.attention(x)
        x = self.norm1(x + attention_out)

        # Feedforward network with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        print(f"Memory after Transformer Layer: {torch.cuda.memory_allocated() / 1e6} MB")
        return x


# Complete LLM Model
class ModularLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, max_seq_len):
        super(ModularLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        self.layers = nn.ModuleList(
            [TransformerLayer(embed_size, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)



# Hyperparameters
vocab_size = 64
embed_size = 128
num_heads = 4
num_layers = 2
hidden_dim = 256
max_seq_len = 50

# Initialize the model and move to GPU
model = ModularLLM(vocab_size, embed_size, num_heads, num_layers, hidden_dim, max_seq_len).cuda()

# Memory tracking
torch.cuda.empty_cache()
print(f"Memory allocated before forward pass: {torch.cuda.memory_allocated() / 1e6} MB")


# Example input
batch_size = 8
seq_len = 50
x = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

def train():
    # Forward pass
    output = model(x)
    print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / 1e6} MB")

    # Loss computation and backward pass
    target = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, vocab_size), target.view(-1))

    print(f"Memory allocated before backward pass: {torch.cuda.memory_allocated() / 1e6} MB")
    loss.backward()
    print(f"Memory allocated after backward pass: {torch.cuda.memory_allocated() / 1e6} MB")

    # Clear gradients
    model.zero_grad()
    print(f"Memory allocated after clearing gradients: {torch.cuda.memory_allocated() / 1e6} MB")

print("Training once to allocate the grads")
train()

print("Doing the real training")
torch.cuda.memory._record_memory_history(max_entries=100000)
train()

path = "/tmp/mem_all.pickle"
torch.cuda.memory._dump_snapshot(path)
print(f"Memory snapshot saved to {path}")
