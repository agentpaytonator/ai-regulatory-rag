import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0]
])

print("Embeddings:")
print(x)
print("Shape:", x.shape)

class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = Q @ K.T

        mask = torch.tril(
            torch.ones(x.shape[0], x.shape[0])
        )

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ V

        return output


head = AttentionHead(4, 2)

output = head(x)

print("Single head output:")
print(output)
print("Single head shape:", output.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            AttentionHead(embedding_size, head_size)
            for i in range(num_heads)
        ])

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        combined = torch.cat(outputs, dim=-1)
        return combined


multi_head = MultiHeadAttention(embedding_size = 4, num_heads = 2, head_size = 2)
output = multi_head(x)
print("Multi-head output:")
print(output)
print("Multi-head shape:", output.shape)