import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

ps = PorterStemmer()
voc_size = 5000
sent_length = 20
embedding_vector_features = 40

df = pd.read_csv('Datasets/FakeNews/train.csv')
df = df.dropna()

x = df.drop("label", axis=1)
y = df['label']

messages = x.copy()
messages.reset_index(inplace=True)

corpus = []
for i in range(0, len(messages)):
    review = re.sub("[^a-zA-Z]", ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

one_hot_repr = [one_hot(words, voc_size) for words in corpus]
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)

model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

x_final = np.array(embedded_docs)
y_final = np.array(y)

print(x_final.shape)
print(y_final.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_final, y_final, test_size=0.33, random_state=42)

model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=10, batch_size=64)

y_pred = model.predict_classes(x_test)

print("Model accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
