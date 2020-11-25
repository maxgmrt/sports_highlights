import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.utils import compute_class_weight
import keras.backend as K
import pandas as pd
import random
import os
import utils
from utils import load_images_from_folder
import numpy as np



def generateModel():
    model = Sequential()
    input_shape=(40, 33, 1)
    
    # Convolutional layer
    model.add(Conv2D(54, (24, 8), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((1, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
        
    # Low-rank layer
    model.add(Dense(32, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    # Dense layer 1
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    
    # Dense layer 2
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    
    # Low-rank layer
    model.add(Dense(1, activation = 'sigmoid'))
    return model


#CALLBACKS AND EARLY STOPPING
def valAccTimesAcc(val_acc, acc):
    return val_acc*acc


def trainModel(nEpochs, gameNames):
    model = generateModel()

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy', valAccTimesAcc])

    MFC_lowlight = []
    MFC_highlight = []

    for g in gameNames:
        MFC_lowlight = MFC_lowlight + load_images_from_folder('mfcc/train/%s_train_l'%(g))
        MFC_highlight = MFC_highlight + load_images_from_folder('mfcc/train/%s_train_h'%(g))

    MFC_train = MFC_highlight + MFC_lowlight

    print('number of total training mfcc: %s' % len(MFC_train))

    X_train = np.asarray(MFC_train)
    X_train = np.array([x.reshape( (40, 33, 1) ) for x in X_train])

    y_train = []
    for i in range(len(MFC_highlight)):
        y_train.append(1)

    for i in range(len(MFC_lowlight)):
        y_train.append(0)

    y_train = np.asarray(y_train)

    classWeight = {0:float(len(MFC_lowlight)/len(MFC_train)), 1:float(len(MFC_highlight)/len(MFC_train))}
    batch_size = 25

    x_train_split, x_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle= True)

    # Early stopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='valAccTimesAcc',
                                  min_delta=0,
                                  patience=5,
                                  verbose=1, mode='max')

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint('model'+'.h5', monitor='valAccTimesAcc', verbose=1, save_best_only=True, mode='max')

    # Train Model
    trained_model = model.fit(x_train_split, y_train_split, shuffle = True, batch_size=20, class_weight = classWeight,
                steps_per_epoch = len(x_train_split) / batch_size, validation_data = (x_val, y_val),
                epochs = nEpochs, callbacks = [checkpoint_callback])

    return trained_model