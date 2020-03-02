# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:35:46 2020

@author: theod
"""

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from tqdm import tqdm
from keras import optimizers

print(tf.__version__)

#CATS V DOGS:

DATADIR = r"C:\Users\theod\Desktop\2K19-20 Modules\Deep Learning\dl-medical-imaging\malaria2"
#setting directory
CATEGORIES_TRAINING = ["Infected_train", "Uninfected_train"]
CATEGORIES_TESTING = ["Infected_test", "Uninfected_test"]

#adjusting image size
IMG_SIZE =140


#creating training data
training_data = []
def create_training_data():
    for category in CATEGORIES_TRAINING:  # do infected & uninfected

        path = os.path.join(DATADIR,category)  # create path to infected & uninfected
        class_num = CATEGORIES_TRAINING.index(category)  # get the classification  (0 or a 1). 0=Infected 1=Uninfected

        for img in tqdm(os.listdir(path)):  # iterate over each image per infected & uninfected
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  #append the array
            except Exception as e: 
                pass
          
create_training_data()
#randomly shuffling the training data           

random.shuffle(training_data)
for sample in training_data[:20]:
    print(sample[1])
print(len(training_data))

x_train = []
y_train = []

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)
 
 #============================================================================= 
#Do the same for test data  
test_data = []
def create_test_data():
    for category in CATEGORIES_TESTING:  # do infected & uninfected

        path = os.path.join(DATADIR,category)  # create path to infected & uninfected
        class_num = CATEGORIES_TESTING.index(category)  # get the classification  (0 or a 1). 0=Infected 1=Uninfected

        for img in tqdm(os.listdir(path)):  # iterate over each image per infected & uninfected
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_test_data()
random.shuffle(test_data)

print(len(test_data))
x_test = []
y_test = []
for sample in test_data[:20]:
    print(sample[1])
for features,label in test_data:
    x_test.append(features)
    y_test.append(label)
     



#==============================================================================
#MNIST FASHION
#==============================================================================
train_images = np.array(x_train)
train_labels = np.array(y_train)
test_images = np.array(x_test)
test_labels = np.array(y_test)
class_names = ['Infected','Uninfected']

train_images.shape
#prints the first image in the training data:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255

test_images = test_images / 255

#since the pixel values range from 0 to 250 we divide by 255 to rescale them from 0 to 1

#Plot the first 35 training images:
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1) 
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)), #Transfprms 2D into 1D i.e lines up pixels
    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(1, activation='sigmoid')#Output (2 probability scores that add up to 1)
    keras.layers.Dense(2, activation = 'softmax')

])

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', #model is updated based on loss and data
              loss='sparse_categorical_crossentropy', #Loss function has to be minimised to steer model
              metrics=['accuracy']) # Monitor training and testing sets

model.fit(train_images, train_labels, epochs=100) #fits model to training data
#One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc) #Comparing loss and accuracy

probability_model = tf.keras.Sequential([model, 
                                        tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

predictions[0]#Shows array of probabilites 
np.argmax(predictions[0])


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
                                100*np.max(predictions_array), #100%
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
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
_ = plt.xticks(range(2), class_names, rotation=45)