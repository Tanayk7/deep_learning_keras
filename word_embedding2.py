# sentiment analysis using embedding layer and pretrainted (glove) embeddings in keras
from numpy import array, asarray, zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

docs = [
    'Well done!',
    'Good work',
    "Great effort",
    'nice work',
    'Excellent',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better'
]

labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# Creates dictionary with all unique words in the vocab
t = Tokenizer()
# Tokenizes the vocab and creates an index
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# Integer encodes the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

glove_embeddings = {}
f = open("word_embeddings/glove.6B.100d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, index in t.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# define the model
model = Sequential()
embedding_layer = Embedding(
    vocab_size, 100, weights=[embedding_matrix], input_length=max_length)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(padded_docs, labels, epochs=50, verbose=2)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=2)
print("Accuracy: ", accuracy*100)
