from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

arabic_characters = ['alif', 'bae', 'tae', 'thae', 'jim', 'hae', 'khaa', 'dal', 'dhal',
                    'rae', 'zain', 'sin', 'chin', 'sad', 'dad', 'tae', 'zain', 'aain',
                    'ghain', 'fae', 'qaf', 'kaf', 'lam', 'mim', 'noun', 'hae', 'waw', 'yae']

len(arabic_characters)
x_train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/csvTrainImages 13440x1024.csv",header=None)
x_test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/csvTestImages 3360x1024.csv",header=None)
y_train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/csvTrainLabel 13440x1.csv",header=None)
y_test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/csvTestLabel 3360x1.csv",header=None)


x_train.head()
x_train.isnull().sum()
x_train.isnull().sum().sum()
x_test.isnull().sum().sum()
y_train.shape
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

y_train.shape
x_train
x_train.shape
y_train.shape
x_test.shape
y_test.shape
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
plt.figure(figsize=(2,2))
print(int(y_train[0]))
plt.imshow(x_train[0].reshape(32,32).T)
plt.xlabel(arabic_characters[int(y_train[0])-1])
plt.figure(figsize=(2,2))
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[0].reshape(32,32).T)
plt.xlabel(arabic_characters[int(y_train[0])-1])
plt.figure(figsize=(2,2))
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[0].reshape(32,32).T,"gray")
plt.xlabel(arabic_characters[int(y_train[0])-1])
ra = np.random.randint(0, 13440, size=25)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[ra[i]].reshape(32,32).T,"gray")
    plt.xlabel(arabic_characters[int(y_train[ra[i]][0])-1])
plt.show()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train.shape
y_train.shape
y_train
len(y_train[0])
y_train = y_train[:,1:]
y_test = y_test[:,1:]
len(y_train[0])

from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(28, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=10)

model.summary()
model.evaluate(x_test,y_test)

from keras.preprocessing.image import ImageDataGenerator
epochs = 100
batch_size = 32 

# construct the training image generator for data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,# set input mean to 0 over the dataset
    samplewise_center=False, # set each sample mean to 0
    featurewise_std_normalization=False, # divide inputs by std of the dataset
    samplewise_std_normalization=False, # divide each input by its std
    zca_whitening=False, # apply ZCA whitening
    rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1, # Randomly zoom image
    width_shift_range=0.1,# randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,# randomly shift images vertically (fraction of total height
    horizontal_flip=False, # randomly flip images
    vertical_flip=False) # randomly flip images

train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
test_gen = datagen.flow(x_test, y_test, batch_size=batch_size)
history = model.fit(train_gen, epochs=epochs,
                             steps_per_epoch=x_train.shape[0]//batch_size,
                             validation_data=test_gen,
                             validation_steps=x_test.shape[0]//batch_size)


plt.figure(figsize=(10,10))
plt.plot(history.history["accuracy"], label='Training accuracy')
plt.plot(history.history["val_accuracy"], label='Validation accuracy')
plt.legend(["accuracy","val_accuracy"])
plt.show()

plt.figure(figsize=(10,10))
plt.plot(history.history["loss"], label='Training loss')
plt.plot(history.history["val_loss"], label='Validation loss')
plt.legend(["loss","val_loss"])
plt.show()

import seaborn as sn
from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset
y_preds = model.predict(x_test)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_preds, axis=1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,10))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

import argparse
import imutils
from imutils.contours import sort_contours
import cv2 as cv
from keras.models import load_model

img = cv.imread("/content/drive/MyDrive/Colab Notebooks/Capture.PNG")
kernel = np.ones((5,5),dtype=np.uint8)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray)
(T, thresh) = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(thresh)
erodation = cv.erode(thresh,kernel,iterations = 1)
plt.imshow(erodation)
blurred = cv.GaussianBlur(erodation, (5,5), 0)
plt.imshow(blurred)
edged = cv.Canny(blurred, 30, 150)
# Since OpenCV 3.2, findContours() no longer modifies the source image.
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]


# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []
plt.imshow(edged)
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv.boundingRect(c)

    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]
        thresh = cv.threshold(roi, 0, 255,
            cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        # if the width is greater than the height, resize along the
        # width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)

        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=32)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)

        # pad the image and force 32x32 dimensions
        padded = cv.copyMakeBorder(thresh, top=dY, bottom=dY,
            left=dX, right=dX, borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0))
        padded = cv.resize(padded, (32, 32))

        # prepare the padded image for classification via our
        # handwriting OCR model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)

        # update our list of characters that will be OCR'd
        chars.append((padded, (x, y, w, h)))
# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
# OCR the characters using our handwriting recognition model
preds = model.predict(chars)
# define the list of label names
labelNames = arabic_characters
# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    secLabel = " %.0f"%(prob * 100)
    label = labelNames[i]+secLabel
   
    # draw the prediction on the image
    # print("{} - {:.2f}%".format(label, prob * 100))
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv.putText(img, label, (x + 7 , y +12), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
   
    # show the image
    #cv.imshow("Image", image)
    #cv.waitKey(0)
    plt.imshow(img,extent=[0, 10000, 0, 10000])
