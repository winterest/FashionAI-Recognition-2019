import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
import keras

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3


from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D

from keras.layers import Input, Flatten, Dense, multiply, Softmax
from keras.models import Model

from keras import optimizers
from keras.optimizers import Adam

from keras import losses

print("TF version is:   ",tf.__version__)

file = open('data/fashionAI_attributes_train1/Annotations/label.csv','r')

paths = []
category = []
label = []
for line in file:
    #print(line)
    #print(line.split(','))
    t = line.split(',')
    paths.append('data/fashionAI_attributes_train1/'+t[0])
    category.append(t[1])
    label.append(t[2][:-1])

categories = list(set(category))
assert len(categories)==8 , "categories not 8"
print("there are 8 categories:", categories)

arr = np.arange(len(paths))
partition = arr
np.random.shuffle(partition)
def label_to_array(llabel):
    return np.array([1 if i=='y' else 0 for i in llabel])

def get_img(index):
    img_path = paths[index]
    img = image.load_img(img_path, target_size=(224,224))
    x_ = image.img_to_array(img)
    x_ = np.expand_dims(x_, axis=0)
    x_ = preprocess_input(x_)    # gives an array in shape of (1,224,224,3)
    label_ = label_to_array(label[index])
    cat_ = categories.index(category[index])
        
    X_cat = np.zeros((80))
    X_cat[cat_*10:cat_*10+10] = np.ones(10)
    
    return x_, label_, cat_, X_cat


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  batch_size=16, dim=(224,224), n_channels=3,
                 n_classes=80, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X_img = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_img = np.empty((self.batch_size,224,224,3))
        y_label = np.zeros((self.batch_size, self.n_classes), dtype=int)
        X_cat = np.zeros((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print(get_img(ID))
            img_,y_,cat_,cat__ =  get_img(ID)
            X_img[i,]=img_
            y_label[i,y_.argmax()+10*cat_]=1

            X_cat[i,] = cat__
            #X_cat[i,cat_*10:cat_*10+10] = np.ones(10)

        return [X_img,X_cat], y_label #keras.utils.to_categorical(y, num_classes=self.n_classes)
    
arr_collar_design_labels = np.arange(8393)

np.random.shuffle(arr_collar_design_labels)

class DataGenerator_arr_collar_design_labels(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  batch_size=16, dim=(224,224), n_channels=3,
                 n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X_img = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_img = np.empty((self.batch_size,224,224,3))
        y_label = np.zeros((self.batch_size, self.n_classes), dtype=int)
        X_cat = np.zeros((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print(get_img(ID))
            img_,y_,cat_,cat__ =  get_img(ID)
            X_img[i,]=img_
            y_label[i,]=y_

        return X_img, y_label #keras.utils.to_categorical(y, num_classes=self.n_classes)
with tf.device('/device:GPU:0'):
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model.name = 'InceptionV3_head'
    
    img_input = Input(shape=(224,224,3),name = 'image_input')

    # add a global spatial average pooling layer
    x = base_model(img_input)
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    my_model = Model(inputs=img_input, outputs=predictions)

EPOCHS = 150
INIT_LR = 0.01
BS = 16

#adm = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
loss = "categorical_crossentropy"
loss_bi = "binary_crossentropy"
loss_mse='mean_squared_error'

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.abs(y_true-y_pred)

my_model.compile(optimizer=sgd, loss=loss_bi, metrics=["accuracy","categorical_accuracy",mean_pred])

# checkpoint
from keras.callbacks import ModelCheckpoint

filepath="incepv3-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
#model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)

training_generator = DataGenerator_arr_collar_design_labels(arr_collar_design_labels[:8000],BS)
validation_generator = DataGenerator_arr_collar_design_labels(arr_collar_design_labels[8000:])

my_model.fit_generator(generator=training_generator,validation_data=validation_generator,shuffle=True,verbose=1,
                       use_multiprocessing=True,epochs=EPOCHS,callbacks=callbacks_list,
                       workers=4)
