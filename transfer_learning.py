import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from glob import glob

IMAGE_SIZE = [224, 224]
INPUT_SHAPE = [224, 224, 3]    # Image dimensions and number of channels

train_path = 'Datasets/DogsVSCats/train'
valid_path = 'Datasets/DogsVSCats/test'

# add preprocessing layer to the front of VGG
vgg16 = VGG16(input_shape=INPUT_SHAPE, weights='imagenet', include_top=False)

# don't train the existing weights
for layer in vgg16.layers:
    layer.trainable = False

folders = glob(train_path + '/*')
x = Flatten()(vgg16.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg16.input, outputs=prediction)

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory(valid_path, target_size=(
    224, 224), batch_size=32, class_mode='categorical')

print("Shape of training set 0th element: ", print(type(training_set)))

r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('TrainedModels/facefeatures_model.h5')
