import torch
import torch.nn as nn

text = "hello hello hello"

vocab = sorted(set(text))
vocab_size = len(vocab)

str_to_int = {character: i for i, character in enumerate(vocab)}
imt_to_str = {i: character for i, character in enumerate(vocab)}

print("Vocabulary:", vocab)
print("Vocabulary size:", vocab_size)

data = torch.tensor([str_to_int[character] for character in text], dtype=torch.long)

print("Encoded data:", data)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    def forward(self, idx):
        logits = self.token_embedding_table(idx)
        return logits

x = data[:-1]
y = data[1:]
    
model = BigramLanguageModel(vocab_size)
logits = model(x)
print("Logits shape:", logits.shape)
print("Logits:", logits)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y)
print("Loss:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for step in range(1000):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Loss after training:", loss.item())