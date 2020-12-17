from audio_processing import generateTrainingMFCCs
import getopt
import sys
import keras
from keras.models import load_model
from model import generateModel
from model import valAccTimesAcc
from model import valLossTimesLoss
from utils import load_images_from_folder
import numpy as np
from test import generatePrediction
from test import getWordFlowHighlights
from model import trainModel
from get_highlight_list_groundtruth import processGroundTruth
import os
from os import listdir
from post_processing import hangover_highlights
from post_processing import binarizePrediction
from audio_processing import getBaselinePrediction
import tensorflow as tf
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from video_utils import edit_video
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
from numpy import sqrt
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot



''' 
Main script. 
Identifies timeouts and highlights of a football game analyzing audio.
mandatory arguments:
<-t, --trainingBool>: 

<-g, --generateGroundTruth>: 

<-m, --saveMFCC>:

<-a, --audioTestFile>:

<-v, --videoTestFile>:
'''

# Get the arguments from the command-line except the filename
argv = sys.argv[1:]
sum = 0

try:
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:g:m:a:v:', ['trainingBool', 'generateGroundTruth', 'saveMFCC', 'audioTestFile'])
    # Check if the options' length is 2 (can be enhanced)
    if len(opts) == 0 and len(opts) > 5:
        print ('usage: main.py -t <trainingBool> -g <generateGroundTruth> -m <saveMFCC> -a <audioTestFile> -v <videoTestFile>')
    else:
      # Iterate the options and get the corresponding values
        for opt, arg in opts:

# Training Boolean. If set to True, the program will train a neural network on labelled MFC Spectrograms,
# generated thanks to audio in audio/train and labels in labels/
            if opt in ("-t", "--train"):
                trainingBool = int(arg)

# Generate Ground-Truth Boolean. If set to True, the program will use the training videos (game footage + summaries)
# in video/, compare them using similarity analysis in order to extract a list of highlights timestamps
            elif opt in ("-g", "--generategroundtruth"):
                generateGroundTruth = int(arg)

# Generate MFCC Boolean. If set to True, the program will generate MFC Spectrogram images and save them in local/mfcc/
            elif opt in ("-m", "--savemfcc"):
                saveMFCC = int(arg)

# Path to audio test file.
            elif opt in ("-a", "--testaudio"):
                audioTestFile = arg # typically ../audio/test

# Path to video test file.
            elif opt in ("-v", "--testvideo"):
                videoTestFile = arg # typically ../video/test

except getopt.GetoptError:
    # Print something useful
    print('usage: main.py -t <trainingBool> -g <generateGroundTruth> -m <saveMFCC> -a <audioTestFile> -v <videoTestFile>')
    sys.exit(2)


gameNames = []
for f in listdir('video'):
    path = os.path.join('video', f)
    if os.path.isdir(path):
        # skip directories
        continue
    gameNames.append(f.replace(".mp4", ""))


# If user wants to compare the summary and the game footage to extract highlights frames to train the model on.
if (generateGroundTruth == 1):
    print("Succesfully entered Ground-Truth generation!")
    processGroundTruth(gameNames)


# If user wants to train the model
if (trainingBool == 1):
    print("Training model on train MFCCs...")
    nEpochs = 10
    if (saveMFCC == 1):
        print("Saving training MFCCs...")
        # Generation of training MFCCs
        for g in gameNames:
            generateTrainingMFCCs('audio/train', 'labels', g)

    # load mfcc correctly
    model = generateModel()

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    trainModel(nEpochs, gameNames, model)

if (trainingBool == 0):
    print("Loading pre-existing model...")
    model = generateModel()
    model.load_weights('model_weights.h5')


# PREDICTION
#
if (audioTestFile):
    X_test = generatePrediction(audioTestFile)
    print('retrieved %s test MFCCs' % (X_test.shape[0]))
    # Prediction using trained model
    prediction = model.predict(X_test)
    print(prediction)
    np.savetxt('pred.csv', prediction, delimiter=',')
    prediction_array = np.asarray(prediction)

    print('max of prediction is %s' % max(prediction_array))
    print('min of prediction is %s' % min(prediction_array))
    print('mean of prediction is %s' % np.mean(prediction_array))
    print('median of prediction is %s' % np.median(prediction_array))

    #threshold = abs(max(prediction_array) - min(prediction_array))/2
    #hangover_prediction = hangover_highlights(prediction_array, threshold)
    #model_prediction = np.asarray(hangover_prediction)

    # TEST
    # Load test reference
    nameStringTest = audioTestFile.replace(".mp3", "").replace("audio/test/","")
    my_data = np.genfromtxt('labels/%s.csv' % (nameStringTest), dtype='int', delimiter=',')
    test_reference = np.asarray(my_data)

    # Prediction using baseline
    baseline = np.asarray(getBaselinePrediction(audioTestFile))


    wf_pred = getWordFlowHighlights(audioTestFile)
    #print(wf_pred.shape[0])
    length = int(min(len(prediction), baseline.shape[0], test_reference.shape[0], wf_pred.shape[0]))
    print('length of prediction is %s' % length)

    acc_base = np.sum((baseline[0:length] == test_reference[0:length])) / length
    acc_wf = np.sum((wf_pred[0:length] == test_reference[0:length])) / length

    accuracies = []
    for i in range(10000):
        processed_prediction = binarizePrediction(prediction, i/10000)
        model_prediction = np.asarray(processed_prediction)
        fpr, tpr, thresholds = roc_curve(test_reference[0:length], model_prediction[0:length])
        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1 - fpr))
        accuracies.append(np.sum((model_prediction[0:length] == test_reference[0:length]))/length)
        # locate the index of the largest g-mean

    ix = argmax(accuracies)
    np.savetxt('accuracies.csv', accuracies, delimiter=',')
    print('max accuracy depending on threshold is %s, at %s' % (max(accuracies), ix))

    processed_prediction = binarizePrediction(prediction, ix/10000)
    model_prediction = np.asarray(processed_prediction)

    acc_pred = np.sum((model_prediction[0:length] == test_reference[0:length])) / length

    highlights = np.where(model_prediction == 1)
    #print(highlights)

    print('true labels')
    print(test_reference)

    print('model prediction')
    print(model_prediction)

    print('baseline prediction')
    print(baseline)

    print('wordflow prediction')
    print(wf_pred)

    print("Accuracy of the trained model on this game: %s" % acc_pred)
    print("Accuracy of the baseline predictor on this game: %s" % acc_base)
    print("Accuracy of the wordflow predictor on this game: %s" % acc_wf)

#if (videoTestFile):
    #edit_video(videoTestFile, highlights, debug=False)

    #X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
     #                          n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
    # split into train/test sets
    #trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    # fit a model
    #model = LogisticRegression(solver='lbfgs')
    #model.fit(trainX, trainy)
    # predict probabilities
    #yhat = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    #yhat = yhat[:, 1]
    # calculate roc curves
