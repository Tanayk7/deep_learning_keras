import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

# Sentences
# One hot representation
# One hot representation to embedding layer keras
# embedding matrix
# vocab_size=10000, dimension=10
vocab_size = 10000
embedding_dimensions = 10
sent_length = 8
sentences = [
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good'
]

print("Sentences: ", sentences)

one_hot_repr = [one_hot(words, vocab_size) for words in sentences]
print(one_hot_repr)

embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dimensions, input_length=sent_length))
model.compile('adam', 'mse')

print(model.summary())

# print(model.predict(embedded_docs))
print(model.predict(embedded_docs)[0])
