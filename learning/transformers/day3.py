import torch

x = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0]
])

print("Token embeddings:")
print(x)
print("Shape:", x.shape)

embedding_size = 3
head_size = 2

query = torch.nn.Linear(embedding_size, head_size, bias=False)
key = torch.nn.Linear(embedding_size, head_size, bias=False)
value = torch.nn.Linear(embedding_size, head_size, bias=False)

Q = query(x)
K = key(x)
V = value(x)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

attention_scores = Q @ K.T

print("Attention scores:")
print(attention_scores)
print("Attention scores:", attention_scores.shape)

mask = torch.tril(torch.ones(3, 3))
masked_scores = attention_scores.masked_fill(
    mask == 0,
    float("-inf")
)
attention_weights = torch.softmax(attention_scores, dim=-1)
print("Attention weights:")
print(attention_weights)
print("Row sums:", attention_weights.sum(dim=-1))
print("Causal mask:")
print(mask)
print("Masked attention weights:")
print(attention_weights)

output = attention_weights @ V
print("Attention output:")
print(output)
print("Output shape:", output.shape)