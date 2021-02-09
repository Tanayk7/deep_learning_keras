# As motivation for this data, let’s suppose that an experimental drug was tested on individuals ranging from age 13 to 100 
# in a clinical trial. The trial had 2100 participants. Half of the participants were under 65 years old, 
# and the other half was 65 years of age or older.

# The trial showed that around 95% of patients 65 or older experienced side effects from the drug, and around 95% of patients 
# under 65 experienced no side effects, generally showing that elderly individuals were more likely to experience side effects.

# Ultimately, we want to build a model to tell us whether or not a patient will experience side effects solely based on the patient's age. 
# The judgement of the model will be based on the training data.

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

test_labels =  []
test_samples = []
train_labels = []
train_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# Create numpy arrays and shuffle the inputs 
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels,train_samples = shuffle(train_labels,train_samples)

# Scale the inputs 
# We reshape the data as a technical requirement just since the fit_transform() function doesn’t accept 1D data by default.
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

# print(train_samples[2000:])
# print(train_labels[2000:])


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

learning_rate = 0.001
batch_size = 10
epochs = 10
log_level = 2 

model = Sequential([
    Dense(units=16,input_shape=(1,), activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=64,activation='relu'),
    Dense(units=2,activation='softmax')
])

print(model.summary())

model.compile(optimizer=Adam(learning_rate=learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(
    x=scaled_train_samples,
    y=train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=log_level,
    validation_split=0.1
)

predictions = model.predict(
    x=scaled_test_samples,
    batch_size=10, 
    verbose=0
)

for i in predictions:
    print(i)

rounded_predictions = np.argmax(predictions,axis=-1)
for i in rounded_predictions:
    print(i)