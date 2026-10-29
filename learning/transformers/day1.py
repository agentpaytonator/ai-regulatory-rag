text = "cringe"

vocab = sorted(set(text))

str_to_int = {character: i for i, character in enumerate(vocab)}

tokens = [str_to_int[character] for character in text]

print("Vocabulary:", vocab)
print("Token IDs:", tokens)

print("Number of tokens in vocabulary: ", len(vocab))

embeddings = {
    0: [0.8, 0.1, -0.3],
    1: [0.2, -0.7, 0.4],
    2: [-0.5, 0.6, 0.9],
    3: [0.3, 0.2, -0.8],
    4: [0.2, -0.9, 0.8],
    5: [0.1, -0.7, -0.5]
}

print("Embedding for 'r': ", embeddings[str_to_int["r"]])

logits = [0.2, -0.5, 3.1, 1.4, 2.2, -0.7]

print("Logits:", logits)

import math

logits =  [0.2, -0.5, 3.1, 1.4, 2.2, -0.7]

expected_values = [math.exp(x) for x in logits]
total = sum(expected_values)

probabilities = [x / total for x in expected_values]

print("Probabilities:", probabilities)
print("Sum:", sum(probabilities))

correct_token_index = 1

loss = -math.log(probabilities[correct_token_index])

print("Cross-entropy loss:", loss)

prediction = 0.2
target = 1.0

loss = (target - prediction) ** 2

print("Loss:", loss)