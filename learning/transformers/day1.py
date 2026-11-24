# Define the input text that will be converted into tokens.
text = "cringe"

# Create a vocabulary for unique characters in alphabetical order.
vocab = sorted(set(text))
# Assign each character a numerical ID, then convert the text into token IDs.
str_to_int = {character: i for i, character in enumerate(vocab)}
tokens = [str_to_int[character] for character in text]

print("Vocabulary:", vocab)
print("Token IDs:", tokens)

# Count the number of unique characters in the vocabulary.
print("Number of tokens in vocabulary: ", len(vocab))

# Assign an embedding, aka numerical vector, to each token ID.
embeddings = {
    0: [0.8, 0.1, -0.3],
    1: [0.2, -0.7, 0.4],
    2: [-0.5, 0.6, 0.9],
    3: [0.3, 0.2, -0.8],
    4: [0.2, -0.9, 0.8],
    5: [0.1, -0.7, -0.5]
}

# Retrieve the embedding corresponding to the character 'r'.
print("Embedding for 'r': ", embeddings[str_to_int["r"]])

# Define logits, which are raw scores for possible next tokens.
# Higher logits correspond to higher probabilities from the softmax function.
logits = [0.2, -0.5, 3.1, 1.4, 2.2, -0.7]

print("Logits:", logits)

import math

# Apply the softmax function to convert logits into probabilities.
logits =  [0.2, -0.5, 3.1, 1.4, 2.2, -0.7]

# Exponentiate each logit.
expected_values = [math.exp(x) for x in logits]
total = sum(expected_values)

# Normalize logits by their total
probabilities = [x / total for x in expected_values]

print("Probabilities:", probabilities)
print("Sum:", sum(probabilities))

# Give the next token a correct index, which is assumed to have index 1.
correct_token_index = 1

# Calculate the cross-entropy loss for the correct next token.
loss = -math.log(probabilities[correct_token_index])
print("Cross-entropy loss:", loss)

prediction = 0.2
target = 1.0

# Calculate the square of the difference to make the error nonnegative.
loss = (target - prediction) ** 2

print("Loss:", loss)