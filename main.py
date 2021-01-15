from audio_processing import generateTrainingMFCCs
import getopt
import sys
import keras
from model import generateModel
import numpy as np
from test import generatePrediction
from test import getWordFlowHighlights
from model import trainModel
from get_highlight_list_groundtruth import processGroundTruth
import os
from os import listdir
from post_processing import binarizePrediction
from audio_processing import getBaselinePrediction
import tensorflow as tf
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from keras_sequential_ascii import keras2ascii
import csv

# GLOBAL VARIABLES
SHOW_ARCH = 1

''' 
Main script. 
Identifies timeouts and highlights of a football game analyzing audio.
mandatory arguments:
<-t, --trainingBool>: 

<-g, --generateGroundTruth>: 

<-m, --saveMFCC>:

<-a, --audioTestFile>:

optional arguments:
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
    print("Succesfully extracted highlights from game footage, see .csv files in labels/ folder")
    sys.exit()

if (saveMFCC == 1):
    print("Saving training MFCCs...")
    # Generation of training MFCCs
    for g in gameNames:
        generateTrainingMFCCs('audio/train', 'labels', g)


# If user wants to train the model
if (trainingBool == 1):
    print("Training model on train MFCCs...")
    nEpochs = 10

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

if (SHOW_ARCH == 1):
    keras2ascii(model)


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
    true_labels = np.asarray(my_data)

    # Prediction using baseline
    baseline = np.asarray(getBaselinePrediction(audioTestFile))

    wf_pred = getWordFlowHighlights(audioTestFile)
    #print(wf_pred.shape[0])
    length = int(min(len(prediction), baseline.shape[0], true_labels.shape[0], wf_pred.shape[0]))
    print('length of prediction is %s' % length)

    acc_base = np.sum((baseline[0:length] == true_labels[0:length])) / length
    acc_wf = np.sum((wf_pred[0:length] == true_labels[0:length])) / length

    accuracies = []
    for i in range(10000):
        processed_prediction = binarizePrediction(prediction, i/10000)
        model_prediction = np.asarray(processed_prediction)
        fpr, tpr, thresholds = roc_curve(true_labels[0:length], model_prediction[0:length])
        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1 - fpr))
        accuracies.append(np.sum((model_prediction[0:length] == true_labels[0:length]))/length)
        # locate the index of the largest g-mean

    ix = argmax(accuracies)
    np.savetxt('accuracies.csv', accuracies, delimiter=',')
    print('max accuracy depending on threshold is %s, at %s' % (max(accuracies), ix))

    processed_prediction = binarizePrediction(prediction, ix/10000)
    model_prediction = np.asarray(processed_prediction)

    acc_pred = np.sum((model_prediction[0:length] == true_labels[0:length])) / length

    highlights = np.where(model_prediction == 1)

    print('true labels')
    print(true_labels)

    print('model prediction')
    print(model_prediction)

    print('baseline prediction')
    print(baseline)

    print('wordflow prediction')
    print(wf_pred)

    with open('predictions_comparison.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(true_labels, model_prediction, baseline, wf_pred))

    print("Accuracy of the trained model on this game: %s" % acc_pred)
    print("Accuracy of the baseline predictor on this game: %s" % acc_base)
    print("Accuracy of the wordflow predictor on this game: %s" % acc_wf)
    
    #PLOTS
    fig, axs = plt.subplots(4)
    hspace = 0.5
    plt.subplots_adjust(hspace=hspace, bottom=0, top=1.5)

    x = np.linspace(0, 18, num=18)
    axs[0].plot(x, true_labels, color='blue')
    axs[1].plot(x, model_prediction, color='red')
    axs[2].plot(x, baseline, color='orange')
    axs[3].plot(x, wf_pred, color='purple')
    axs[0].title.set_text('True Labels')
    axs[1].title.set_text('Model Prediction')
    axs[2].title.set_text('Energy-based Predictor')
    axs[3].title.set_text('Wordflow Predictor')