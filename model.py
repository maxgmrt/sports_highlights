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
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from sklearn.utils import class_weight


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


def trainModel(nEpochs, gameNames, model):
    MFC_lowlight = []
    MFC_highlight = []

    for g in gameNames:
        MFC_lowlight = MFC_lowlight + load_images_from_folder('mfcc/train/%s_l'%(g))
        MFC_highlight = MFC_highlight + load_images_from_folder('mfcc/train/%s_h'%(g))

    MFC_train = MFC_highlight + MFC_lowlight

    print('number of total training mfcc: %s' % len(MFC_train))

    X_train = np.asarray(MFC_train)
    X_train = np.array([x.reshape( (40, 33, 1) ) for x in X_train])

    y_train = []
    for i in range(len(MFC_highlight)):
        y_train.append(1)

    for i in range(len(MFC_lowlight)):
        y_train.append(0)

    print(len(y_train))
    zeros = 0
    ones = 0

    for i in range(len(y_train)):
        if y_train[i] == 0:
            zeros = zeros + 1
        if y_train[i] == 1:
            ones = ones + 1

    print("this is th enimber of ones %s"% ones)
    print("this is th enimber of eros %s"% zeros)


    y_train = np.asarray(y_train)
    print(1/float(len(MFC_lowlight)/len(MFC_train)))
    print(1/float(len(MFC_highlight)/len(MFC_train)))
    #classWeight = {0:1/float(len(MFC_lowlight)/len(MFC_train)), 1:1/float(len(MFC_highlight)/len(MFC_train))}


    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {i: class_weights[i] for i in range(2)}
    print(class_weights)

    print('class distribution is %s / %s' % (float(len(MFC_lowlight)/len(MFC_train)), float(len(MFC_highlight)/len(MFC_train))))
    batch_size = 40

    x_train_split, x_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle= True)
    print()
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint('model'+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    # Train Model
    model.fit(x_train_split, y_train_split, shuffle = True, batch_size=batch_size, class_weight = class_weights, steps_per_epoch = int(len(x_train_split) / batch_size), validation_data = (x_val, y_val), epochs = nEpochs, callbacks = [checkpoint_callback])

    model.save_weights('model_weights.h5')
    return