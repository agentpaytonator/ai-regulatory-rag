# Import PyTorch to work with tensors and train neural networks.
import torch
# torch.nn contains tools for building neural network models.
import torch.nn as nn

# Define the training text that the model will learn from.
text = "hello hello hello"

# Create a vocabulary for unique characters in alphabetical order.
vocab = sorted(set(text))
# vocab_size represents the number of possible tokens.
vocab_size = len(vocab)

# Create dictionaries that convert characters into token IDs and vice versa.
# str_to_int converts characters into token IDs.
str_to_int = {character: i for i, character in enumerate(vocab)}
# imt_to_str converts token IDs into characters.
imt_to_str = {i: character for i, character in enumerate(vocab)}

print("Vocabulary:", vocab)
print("Vocabulary size:", vocab_size)

# Convert each character in the training text into a token ID.
# torch.tensor stores these IDs in a PyTorch tensor.
# dtype=torch.long creates integer values, which can be used to lookup embeddings.
data = torch.tensor([str_to_int[character] for character in text], dtype=torch.long)

print("Encoded data:", data)

# Define a bigram language model using PyTorch's nn.Module.
# A bigram model predicts the next token using only the current token.
class BigramLanguageModel(nn.Module):
    # Initialize the model and create a trainable embedding table.
    def __init__(self, vocab_size):
        super().__init__()
        # The table has vocab_size rows and vocab_size columns.
        # Each row contains raw prediction scores for the next token.
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    # Define how the model processes input token IDs.
    def forward(self, idx):
        # Look up each token's row in the embedding table.
        logits = self.token_embedding_table(idx)
        # The resulting logits are scores for possible next tokens.
        return logits

# x contains every token except the last.
x = data[:-1]
# y contains every token except the first.
y = data[1:]

# Create the bigram model and make initial predictions.  
model = BigramLanguageModel(vocab_size)
# The logits tensor has shape [number_of_input_tokens, vocab_size].
logits = model(x)
print("Logits shape:", logits.shape)
print("Logits:", logits)

# Declare a loss function that measures how well the model predicts the correct next token.
loss_fn = nn.CrossEntropyLoss()
# Calculate cross-entropy loss between the predicted logits and correct tokens.
loss = loss_fn(logits, y)
print("Loss:", loss.item())

# Create an AdamW optimizer to update the model's trainable parameters.
# The learning rate controls the size of each parameter update.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Train the model for 1000 iterations.
# Each iteration makes predictions, measures error, and updates parameters.
for step in range(1000):

    # Predict the next-token logits.
    logits = model(x)

    # Calculate the difference between the predictions and the correct targets.
    loss = loss_fn(logits, y)

    # Gradients indicate how the parameters affect the loss.
    # Clear gradients left from the previous training iteration.
    optimizer.zero_grad()
    # Caculate gradients of the loss backwards.
    loss.backward()

    # Update model parameters using the calculated gradients.
    optimizer.step()

# Display the loss calculated during the final training iteration.
print("Loss after training:", loss.item())