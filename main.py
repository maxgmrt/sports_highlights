from audio_processing import generateTrainingMFCCs
from audio_processing import generateMFCC
from post_processing import hangover_highlights
import getopt
import sys
import keras
from utils import load_images_from_folder
import numpy as np
from model import trainModel
from get_highlight_list_groundtruth import processGroundTruth
import os
from os import listdir

''' 
Main script. 
Identifies timeouts and highlights of a football game analyzing audio.
mandatory arguments:
<-t, --trainingBool>: 

<-g, --generateGroundTruth>: 

<-m, --saveMFCC>:

<-a, --audioTestFile>:
'''

# Get the arguments from the command-line except the filename
argv = sys.argv[1:]
sum = 0

try:
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:g:m:a:', ['trainingBool', 'generateGroundTruth', 'saveMFCC', 'audioTestFile'])
    # Check if the options' length is 2 (can be enhanced)
    if len(opts) == 0 and len(opts) > 5:
        print ('usage: main.py -t <trainingBool> -g <generateGroundTruth> -m <saveMFCC> -a <audioTestFile>')
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
                audioTestFile = arg # typically ../audio/train

except getopt.GetoptError:
    # Print something useful
    print('usage: main.py -a <audio_path> -v <videos_path> -n <name_of_game> -t <training_parameter> -g <generateGroundTruth>')
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


if (trainingBool == 1):
    nEpochs = 30
    if (saveMFCC == 1):
        # Generation of training MFCCs
        for g in gameNames:
            generateTrainingMFCCs('audio/train', 'labels', g)

    # load mfcc correctly
    trained_model = trainModel(nEpochs, gameNames)

else:
    trained_model = keras.models.load_model('model.h5')

# PREDICTION
#
#
#
if (audioTestFile):
    generateMFCC(audioTestFile, audioTestFile.replace(".mp3", ""),  [], [], macro=False, test=True)
    MFC_test = load_images_from_folder('mfcc/test/%s'%(audioTestFile.replace(".mp3", "")))

    X_test = np.asarray(MFC_test)
    X_test = np.array([x.reshape( (40, 33, 1) ) for x in X_test])
    y_test = np.ones(len(MFC_test))

    trained_model = keras.models.load_model('model.h5')
    prediction = trained_model.predict(X_test)
    prediction = np.asarray(prediction)
    hangover_prediction = hangover_highlights(prediction)

print(hangover_prediction)


#getBaselinePrediction(audMonoGame, sample_rate)
