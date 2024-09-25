
# Animal Image Detection System

## Overview
This project implements an image detection system using Convolutional Neural Networks (CNN) to classify images of different animals. The model is trained to recognize four classes: Bears, Crows, Elephants, and Rats.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Prediction](#prediction)
- [License](#license)

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib (optional for visualization)
- Google Colab (optional for cloud execution)

## Dataset
The dataset consists of images categorized into training and testing directories:
- `Training`: Contains images for training the model.
- `Testing`: Contains images for evaluating the model's performance.

Make sure to unzip the dataset using:
```bash
!unzip 'Animal_Dataset.zip'
```

## Installation
To set up the environment, you can run the following commands in a Google Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
1. **Data Augmentation**: The training images are augmented using the `ImageDataGenerator` class from Keras, which applies rescaling, zooming, and horizontal flipping.
2. **Model Building**: A Sequential CNN model is constructed with:
   - Convolutional layer
   - Max pooling layer
   - Flattening layer
   - Fully connected layers with ReLU activation
   - Output layer with softmax activation for classification

3. **Training**: The model is trained using the training dataset for a specified number of epochs.

4. **Prediction**: Once trained, the model can predict the class of new images.

## Model Architecture
The CNN model consists of the following layers:
- **Convolution Layer**: 32 filters, kernel size (3,3), ReLU activation
- **Max Pooling Layer**: Pool size (2,2)
- **Flatten Layer**
- **Dense Layers**:
  - Hidden layer with 300 units and ReLU activation
  - Hidden layer with 150 units and ReLU activation
  - Output layer with 4 units (for 4 classes) and softmax activation

## Training
The model is trained with the following parameters:
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy

Training is done using the `fit_generator` method:
```python
model.fit_generator(xtrain, steps_per_epoch=len(xtrain), epochs=10, validation_data=xtest, validation_steps=len(xtest))
```

## Prediction
To make predictions on new images, use the following code snippet:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_your_image.jpeg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = np.argmax(model.predict(x))
op = ['bears', 'crows', 'elephants', 'rats']
print(op[pred])
```

## License
This project is licensed under the MIT License .
