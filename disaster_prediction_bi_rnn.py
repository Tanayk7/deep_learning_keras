 import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

stopset = stopwords.words("english")
punctuation = string.punctuation
tweet_tokenizer = TweetTokenizer()
t = Tokenizer()
embedding_vector_features = 100
sent_length = None
voc_size = None

def remove_stopwords(sentence):
    clean_sentence = " ".join([word for word in sentence if word not in stopset])
    return clean_sentence

def remove_hashtags(sentence):
    for word in sentence.split():
        if(word[0] == '#'):
            word = word.replace("#", "")
    return sentence

def remove_mentions(sentence):
    for word in sentence.split():
        if(word[0] == '@'):
            word = word.replace("@", "")
    return sentence

def remove_links(sentence):
    clean_sentence = " ".join([word for word in sentence.split() if "http" not in word])
    return clean_sentence

def preprocess_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        tokenized_tweets = [word.lower() for word in tweet_tokenizer.tokenize(tweet) if word.isalnum()]
        tokenized_tweets = list(filter(lambda token: token not in string.punctuation, tokenized_tweets))
        clean_tweet = remove_stopwords(tokenized_tweets)
        clean_tweet = remove_hashtags(clean_tweet)
        clean_tweet = remove_links(clean_tweet)
        clean_tweet = remove_mentions(clean_tweet)
        clean_tweets.append(clean_tweet)
    return clean_tweets

def get_max_sent_len(tweets):
    max_length = len(tweets[0].split())
    for tweet in tweets: 
        tweet_length = len(tweet.split())
        if(tweet_length > max_length):
            max_length = tweet_length
    return max_length

train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv("Datasets/test.csv")
train_list = list(train_df['text'])
test_list = list(test_df['text'])

# preprocess the text 
train_tweets = preprocess_tweets(train_list)
test_tweets = preprocess_tweets(test_list)

# get maximum sentence length
corpus = train_tweets + test_tweets
sent_length = get_max_sent_len(corpus)
print("Maximum sentence length: ",sent_length)

# get total vocab size 
t.fit_on_texts(corpus)
voc_size = len(t.word_index) + 1
print("Vocab length: ",voc_size)

# load glove embeddings
glove_embeddings = {}
f = open("word_embeddings/glove.6B.100d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((voc_size, 100))
for word, index in t.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# create one hot representations and then embedded word vectors 
one_hot_repr_train = [one_hot(words, voc_size) for words in train_tweets]
embedded_docs_train = pad_sequences(one_hot_repr_train, padding='pre', maxlen=sent_length)
one_hot_repr_test = [one_hot(words, voc_size) for words in test_tweets]
embedded_docs_test = pad_sequences(one_hot_repr_train, padding='pre', maxlen=sent_length)

# Create inputs for the model 
x_final = np.array(embedded_docs_train)
y_final = train_df[['target']].to_numpy().astype("float32")
x_test = np.array(embedded_docs_test)

# create model 
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,weights=[embedding_matrix],input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(units=1,kernel_initializer='he_uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

model.fit(x_final, y_final, validation_split=0.3, epochs=100, batch_size=50)
y_pred = model.predict_classes(x_test)

print(y_pred > 0.5)