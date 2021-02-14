# Kaggle Disaster tweet prediction challenge 
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

stopset = stopwords.words("english")
punctuation = string.punctuation
tweet_tokenizer = TweetTokenizer()
word_embeddings = {}


def remove_stopwords(sentence):
    clean_sentence = " ".join(
        [word for word in sentence if word not in stopset])
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
    clean_sentence = " ".join(
        [word for word in sentence.split() if "http" not in word])
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

def load_word_embeddings(word_embeddings):
    f = open('word_embeddings/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

def get_tweet_vectors(tweets,word_embeddings):
    tweet_vectors = []
    for tweet in tweets:
        if(len(tweet) != 0):
            sentence_vector = [word_embeddings.get(word, np.zeros((100,))) for word in tweet.split()]
            normalized_vector = sum(sentence_vector) / (len(sentence_vector) + 0.001)
            tweet_vectors.append(normalized_vector)
        else:
            tweet_vectors.append(np.zeros(100,))
    return np.asarray(tweet_vectors)

# read the data
train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv("Datasets/test.csv")
train_list = list(train_df['text'])
test_list = list(test_df['text'])
# Prepare TRAIN SET
train_tweets = preprocess_tweets(train_list)
train_labels = train_df[['target']].to_numpy().astype("float32")
# Prepare TEST SET
test_tweets = preprocess_tweets(test_list)

# extract word embeddings 
load_word_embeddings(word_embeddings)

# generate sentence vectors for train and test tweets 
train_tweets = get_tweet_vectors(train_tweets,word_embeddings)
test_tweets = get_tweet_vectors(test_tweets,word_embeddings)

# build the model 
model = Sequential()
model.add(Dense(input_dim=100,units=250,kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=500,kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=250,kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=250,kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model_history = model.fit(train_tweets,train_labels,validation_split=0.2,batch_size=50, epochs=100)
model.save("disaster_prediction_model")

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_set', 'validation_set'], loc='upper left')
plt.show()

y_pred = model.predict(test_tweets)
y_pred = (y_pred > 0.5)

f = open("submission.csv",'w')
f.write("id,target\n")
for i in range(len(y_pred)):
    # print("tweet: {}, predicted: {}".format(test_df['text'][i],('disaster' if y_pred[i][0] == True else 'not disaster')))
    line = str(test_df['id'][i]) + "," + str(int(y_pred[i][0])) + "\n"
    f.write(line)
f.close()


