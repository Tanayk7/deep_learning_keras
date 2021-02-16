from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, ELU


dataset = pd.read_csv('Datasets/churn_modelling.csv')
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Create dummy variables
geography = pd.get_dummies(x['Geography'], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)

# Concatenate the dataframe
x = pd.concat([x, geography, gender], axis=1)
# Drop unnecessary columns
x = x.drop(['Geography', 'Gender'], axis=1)

print(x.head())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
print("training_data: ", x_train[len(x_train)-3:])
print(type(x_train))
print(x_train.shape)
x_test = sc.fit_transform(x_test)

# Create the model
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='he_uniform',
                     activation='relu', input_dim=11))
classifier.add(
    Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(
    Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='Adamax',
                   loss='binary_crossentropy', metrics=['accuracy'])
model_history = classifier.fit(
    x_train, y_train, validation_split=0.33, batch_size=50, epochs=100)

print(model_history.history.keys())

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_set', 'validation_set'], loc='upper left')
plt.show()

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
score = accuracy_score(y_pred, y_test)
print("Accuracy score: ", score)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Calculate the accuracy


plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
