# Import PyTorch to work with tensors operations and neural networks components.
import torch
# torch.nn provides layers and classes for building neural networks.
import torch.nn as nn

# Define embeddings for 3 tokens, each represented by 4 numbers.
# Each row represents one token, and each column is an embedding dimension.
# The input tensor shape is (3, 4).
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0]
])

# Display input embeddings and their dimensions.
print("Embeddings:")
print(x)
print("Shape:", x.shape)

# Define a single self-attention head as a PyTorch neural network module.
class AttentionHead(nn.Module):
    # Initialize the attention head.
    # embedding_size = number of features in each input token.
    # head_size = number of features in each Q, K, and V vector.
    def __init__(self, embedding_size, head_size):
        super().__init__()

        # Create 3 independent linear transformations.
        # Each maps an input embedding to a smaller head-sized vector.
        # bias=False means no additional bias term is added.
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    # Define the calculations performed when this head receives input.
    # PyTorch automatically calls forward() when we use head(x).
    def forward(self, x):

        # Input embeddings turn into Query, Key, Value vectors.
        # Q: What each token is looking for.
        Q = self.query(x)
        # K: What each token can be matched against.
        K = self.key(x)
        # V: What information each token can contribute.
        V = self.value(x)

        # Calculate attention scores using Q * (K transpose).
        # Q has shape (3, 2), and K.T has shape (2, 3).
        # The result is a (3, 3) matrix of token-to-token scores.
        attention_scores = Q @ K.T

        # Create a causal mask using a lower-triangular matrix.
        # x.shape[0] is the number of tokens (Here we have 3 tokens).
        # A 1 allows attention to future tokens 
        # A 0 blocks(prohibits) attention to future tokens.
        mask = torch.tril(
            torch.ones(x.shape[0], x.shape[0])
        )

        # Replace blocked scores with negative infinity.
        # That will make all future token positions receive no attention.
        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # The softmax function converts masked attention scores into attention weights.
        # Nonnegative weights sum up to 1 in each row.
        # dim=-1 applies softmax over the entire last dimension.
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        # Multiply attention weights by the Value matrix V.
        # (3, 3) @ (3, 2) produces an output shape (3, 2)
        output = attention_weights @ V

        # Return the context-aware representations for this head.
        return output


# Create an attention head with embedding size 4 and head size 2.
# The head accepts 4-dimensional inputs and outputs 2-dimensional vectors.
head = AttentionHead(4, 2)

# Run the input embeddings through the single attention head.
output = head(x)

# Display the output.
print("Single head output:")
print(output)

# Display the dimensions.
print("Single head shape:", output.shape)

# Define a multi-head attention module.
# Multiple independent attention heads process the same input in parellel.
class MultiHeadAttention(nn.Module):

    # Initialize the multi-head attention module.
    # embedding_size = input embedding dimensions.
    # num_heads = number of independent attention heads.
    # head_size = output dimensions produced by each head.
    def __init__(self, embedding_size, num_heads, head_size):
        super().__init__()

        # By using a for loop, create multiple independent attention heads,
        # each with its own Q, K, and V weights.
        self.heads = nn.ModuleList([
            AttentionHead(embedding_size, head_size)
            for i in range(num_heads)
        ])

    # Define how multiple attention heads process the input.
    def forward(self, x):

        # Loop through each attention head and process the same input embeddings independently.
        outputs = [head(x) for head in self.heads]

        # Concatenate the outputs from all attention heads along the last dimension.
        # For 2 heads, combine two (3, 2) matrices into one (3, 4) matrix.
        combined = torch.cat(outputs, dim=-1)

        # Return the combined multi-head attention output.
        return combined

# Create a multi-head attention module with:
# 4 input features per token, 2 heads, and 2 features per head.
multi_head = MultiHeadAttention(embedding_size = 4, num_heads = 2, head_size = 2)

# Process the input through both attention heads.
output = multi_head(x)

# Display the combined output.
print("Multi-head output:")
print(output)

# Display the dimensions of the multi-head attention.
print("Multi-head shape:", output.shape)