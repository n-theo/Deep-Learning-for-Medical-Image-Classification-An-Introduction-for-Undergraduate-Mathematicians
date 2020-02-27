# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:29:53 2019

@author: theod
"""

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist #import dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #training set vs test set
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images.shape
#prints the first image in the training data:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

#since the pixel values range from 0 to 250 we divide by 255 to rescale them from 0 to 1

#Plot the first 35 training images:
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1) 
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabelclass_names[train_labels[i]])
plt.show()(


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #Transfprms 2D into 1D i.e lines up pixels
    keras.layers.Dense(128, activation='relu'),#
    keras.layers.Dense(10, activation='softmax')#Output (10 probability scores that add up to 1)

])

model.compile(optimizer='adam', #model is updated based on loss and data
              loss='sparse_categorical_crossentropy', #Loss function has to be minimised to steer model
              metrics=['accuracy']) # Monitor training and testing sets

model.fit(train_images, train_labels, epochs=10) #fits model to training data
#One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc) #Comparing loss and accuracy

predictions = model.predict(test_images)
#predictions[0] #Shows array of probabilites 
#np.argmax(predictions[0])
#

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 0 #0th image, correect prediction is blue, incoreect is red
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


#Predictions for the first 15 images:
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#Prediction for image in test data:



img = test_images[4]
print(class_names[test_labels[4]])
plt.figure()
plt.imshow(test_images[4])
plt.colorbar()
plt.grid(False)
plt.show()

print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
