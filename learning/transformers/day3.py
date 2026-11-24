import torch

# Define embeddings for 3 tokens, with each token represented as an 3-entry row vector.
# Each row represents one token, and each column is an embedding dimension.
x = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0]
])

# Display the embeddings and their dimensions (3 tokens x 3 values)
print("Token embeddings:")
print(x)
print("Shape:", x.shape)

# Define the embedding size as the number of values representing each input token.
embedding_size = 3
# Define the attention head size as the number of values in each Query, Key, and Value vector.
head_size = 2

# Create 3 separate linear transformations for Query, Key, and Value.
# Each transformation maps a 3-dimensional embedding to 2 dimensions.
# bias=False means no additional bias term is added.
# These weights are randomly initialized and would be learned during training.
query = torch.nn.Linear(embedding_size, head_size, bias=False)
key = torch.nn.Linear(embedding_size, head_size, bias=False)
value = torch.nn.Linear(embedding_size, head_size, bias=False)

# Apply the linear transformations to the input embeddings.
# Q (Query): what each token is looking for.
Q = query(x)
# K (Key): what each token can be matched against.
K = key(x)
# V (Value): the information each token can contribute.
V = value(x)

# Each matrix has shape (3, 2): 3 tokens, 2 values per token.
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Calculate attention scores by multiplying Q by the transpose of K.
# Each score measures the learned alignment between two tokens.
# Q has shape (3, 2), and K.T has shape (2, 3).
# Therefore, attention_scores has shape (3, 3).
attention_scores = Q @ K.T

print("Attention scores:")
print(attention_scores)
print("Attention scores:", attention_scores.shape)

# Create a causal mask using a lower-triangular matrix.
# A value of 1 means the token is allowed to attend to that position.
# A value of 0 means the position is blocked.
# This prevents tokens from attending to future tokens.
mask = torch.tril(torch.ones(3, 3))

# Replace blocked attention scores with negative infinity.
# After softmax, these positions will have probability 0.
masked_scores = attention_scores.masked_fill(
    mask == 0,
    float("-inf")
)

# Apply softmax to the MASKED scores.
# Softmax converts attention scores into nonnegative attention weights.
# dim=-1 means softmax is applied across each row.
# Each row of attention weights should sum to approximately 1.
attention_weights = torch.softmax(attention_scores, dim=-1)

print("Attention weights:")
print(attention_weights)
print("Row sums:", attention_weights.sum(dim=-1))
print("Causal mask:")
print(mask)
print("Masked attention weights:")
print(attention_weights)

# Multiply the attention weights by the Value matrix.
# Each output row is a weighted combination of Value vectors.
# This produces a context-aware representation for each token.
# (3, 3) @ (3, 2) gives an output shape of (3, 2).
output = attention_weights @ V

print("Attention output:")
print(output)
print("Output shape:", output.shape)