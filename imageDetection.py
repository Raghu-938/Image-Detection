# Import necessary libraries
import os
from zipfile import ZipFile
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing import image

# Unzip the dataset
with ZipFile('/content/Animal_Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

# Mount Google Drive for access
drive.mount('/content/drive')

# Data augmentation on training variable
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
xtrain = train_datagen.flow_from_directory('/content/dataset/Training',
                                           target_size=(64, 64),
                                           class_mode='categorical',
                                           batch_size=100)

# Load testing data
xtest = test_datagen.flow_from_directory('/content/dataset/Testing',
                                         target_size=(64, 64),
                                         class_mode='categorical',
                                         batch_size=100)

# Build a CNN model
model = Sequential()  # Initializing sequential model
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # Convolution layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Flatten())  # Flatten layer
model.add(Dense(300, activation='relu'))  # Hidden layer 1
model.add(Dense(150, activation='relu'))  # Hidden layer 2
model.add(Dense(4, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(xtrain,
          steps_per_epoch=len(xtrain),
          epochs=10,
          validation_data=xtest,
          validation_steps=len(xtest))

# Save the model
model.save('animal.h5')

# Load and preprocess an image for prediction
img = image.load_img('/content/dataset/Testing/elephants/photo_1552055570_5c41ef975579.jpeg', target_size=(64, 64))  # Reading image
x = image.img_to_array(img)  # Converting image into array
x = np.expand_dims(x, axis=0)  # Expanding dimensions
pred = np.argmax(model.predict(x))  # Predicting the highest probability index

# Create a list of class names
op = ['bears', 'crows', 'elephants', 'rats']  # Class names
result = op[pred]  # Output the predicted class
print(f'The predicted class is: {result}')
