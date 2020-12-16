import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
#from keras import optimizers
#from keras.optimizers import Adam
#from keras.callbacks import Callback
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
from utils import load_pickle_from_folder
import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from sklearn.utils import class_weight
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import random
import tensorflow as tf


def generateModel():
    model = Sequential()
    input_shape=(40, 33, 1)
    
    # Convolutional layer
    model.add(Conv2D(54, (24, 8), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((1, 8), strides=(1, 1)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
        
    # Low-rank layer
    model.add(Dense(32, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.7))

    # Dense layer 1
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.7))
    
    # Dense layer 2
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.7))
    
    # Low-rank layer
    model.add(Dense(1, activation = 'sigmoid'))
    return model


#CALLBACKS AND EARLY STOPPING
def valAccTimesAcc(val_acc, acc):
    return val_acc * acc

def valLossTimesLoss(val_loss, loss):
    return (loss)

def trainModel(nEpochs, gameNames, model):
    MFC_lowlight = []
    MFC_highlight = []

    for g in gameNames:
        print('Loading MFC Spectrograms from game %s' % g)
        MFC_highlight = MFC_highlight + load_pickle_from_folder('mfcc/train/%s_h' % g)
        MFC_lowlight = MFC_lowlight + load_pickle_from_folder('mfcc/train/%s_l' % g)


    print('Successfully loaded %s highlight MFC spectrograms of dimensions %s x %s' % (len(MFC_highlight), MFC_highlight[0].shape[0], MFC_highlight[0].shape[1]))
    print('Successfully loaded %s lowlight MFC spectrograms of dimensions %s x %s' % (len(MFC_lowlight), MFC_lowlight[0].shape[0], MFC_lowlight[0].shape[1]))

    MFC_train = MFC_highlight + MFC_lowlight

    print('number of total training mfcc: %s' % len(MFC_train))

    X_train = np.asarray(MFC_train)
    X_train = np.array([x.reshape((40, 33, 1)) for x in X_train])

    y_train = []
    for i in range(len(MFC_highlight)):
        y_train.append(1)

    for i in range(len(MFC_lowlight)):
        y_train.append(0)

    print(len(y_train))

    y_train = np.asarray(y_train)
    print(1/float(len(MFC_lowlight)/len(MFC_train)))
    print(1/float(len(MFC_highlight)/len(MFC_train)))
    #classWeight = {0:1/float(len(MFC_lowlight)/len(MFC_train)), 1:1/float(len(MFC_highlight)/len(MFC_train))}


    batch_size = 32
    indices = np.arange(len(MFC_train))
    #x_train_split, x_val, y_train_split, y_val, idx_split, idx_val = train_test_split(X_train, y_train, indices, test_size=0.2) #shuffle= True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=(random.randint(0,len(MFC_train))))
    sss.get_n_splits(X_train, y_train)

    for train_index, val_index in sss.split(X_train, y_train):
        print("TRAIN:", train_index, "TEST:", val_index)
        x_train_split, x_val = X_train[train_index], X_train[val_index]
        y_train_split, y_val = y_train[train_index], y_train[val_index]

    #print('y_train cropped is %s' % y_train[idx_split])
    print('proportion is %s' % ((sum(y_train_split)/sum(y_train))/(2*0.8)))
    print('proportion is %s' % ((sum(y_val)/sum(y_train))/(2*0.2)))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_split), y_train_split)
    class_weights = {i: class_weights[i] for i in range(2)}
    print(class_weights)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint('model'+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1, mode='min',)
    # Train Model class_weight=class_weights
    model.fit(x_train_split, y_train_split, shuffle=True, batch_size=batch_size, class_weight=class_weights,
              steps_per_epoch=int(len(x_train_split) / batch_size), validation_data=(x_val, y_val),
              epochs=nEpochs, callbacks=[earlystop], verbose = 1)

    model.save_weights('model_weights.h5')
    return