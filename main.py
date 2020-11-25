import get_highlight_list_groundtruth
from get_highlight_list_groundtruth import get_highlight_list
import utils
import model
from audio_processing import generateTrainingMFCCs
import post_processing
from post_processing import hangover_highlights
import getopt
import sys
import utils
import keras
from utils import load_images_from_folder
from utils import frameToTimestamp
import numpy as np
import cv2
import librosa
from model import trainModel

''' 
Main script. 
Identifies timeouts and highlights of a football game analyzing audio.
mandatory arguments:
<-a, --audio>: defines the path to the audio folder

<-v, --video>: defines the path to the video files

<-n, --name>: defines the name of the game (ex: PSGvSCO)

<-t, --train>: defines if training needs to be performed
'''


# Get the arguments from the command-line except the filename
argv = sys.argv[1:]
sum = 0

try:
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:g:a:', ['trainingBool', 'generateGroundTruth', 'audioTestFile'])
    # Check if the options' length is 2 (can be enhanced)
    if len(opts) == 0 and len(opts) > 5:
        print ('usage: main.py -t <trainingBool> -g <generateGroundTruth> -a <audioTestFile>')
    else:
      # Iterate the options and get the corresponding values
        for opt, arg in opts:

# Training Boolean. If set to True, the program will train a neural network on labelled MFC Spectrograms,
# generated thanks to audio in audio/train and labels in labels/
            if opt in ("-t", "--train"):
                trainingBool = arg

# Generate Ground-Truth Boolean. If set to True, the program will use the training videos (game footage + summaries)
# in video/train, compare them using similarity analysis in order to extract a list of highlights timestamps
            elif opt in ("-g", "--generategroundtruth"):
                generateGroundTruth = arg

# Path to audio test file.
            elif opt in ("-a", "--testaudio"):
                audioTestFile = arg # typically ../audio/train

except getopt.GetoptError:
    # Print something useful
    print('usage: main.py -a <audio_path> -v <videos_path> -n <name_of_game> -t <training_parameter> -g <generateGroundTruth>')
    sys.exit(2)

nEpochs = 30
framesPath = 'video/%s_frames'%(nameString) # typically ../video/PSGvSCO_frames/
labelsPath = 'labels'

# LOADING THE AUDIO
#audMonoGame, sample_rate = librosa.load(audioPath, sr=16000, mono=True)

# If user wants to compare the summary and the game footage to extract highlights frames to train the model on.
if (generateGroundTruth == 1):
    # DELEGUER A UN SOUS-FICHIER !
    overwrite=True
    every=100
    chunk_size=1000

    gamePath = '%s/%s.mp4' % (videosPath, nameString)
    sumPath = '%s/%s_sum.mp4' % (videosPath, nameString)
    print(gamePath)
    print(sumPath)
    cap = cv2.VideoCapture(gamePath)
    fps_game = cap.get(cv2.CAP_PROP_FPS)
    print(fps_game)
    cap = cv2.VideoCapture(sumPath)
    fps_sum = cap.get(cv2.CAP_PROP_FPS)
    print(fps_sum)
    # Returns the frame numbers of the highlights in the original game footage
    highlights_frame_number = get_highlight_list(gamePath, sumPath, framesPath, overwrite, every, chunk_size)
    # and the timestamps
    highlights_timestamps = frameToTimestamp(highlights_frame_number, fps_game)

    # ADD THE LOWLIGHT PART !
    lowlight_timestamps = []

    highlights_timestamps_train = highlights_timestamps
    lowlights_timestamps_train = lowlight_timestamps

highlights_timestamps_test = []
lowlights_timestamps_test = []

if (trainingBool):
    audioTrainPath = '%s/train'% (audiosPath)
    generateTrainingMFCCs(audioTrainPath, labelsPath, nameString)
    #generateMFCC(audioPath, nameString, highlights_timestamps = [], lowlights_timestamps = [], macro=True, test=False)
    #generateMFCC(audiosPath, nameString, highlights_timestamps_test, lowlights_timestamps_train, macro=False, test=True)
    #generateMFCC(audioPath, nameString, highlights_timestamps = [], lowlights_timestamps = [], macro=True, test=True)
# TRAINING
    trained_model = trainModel(nEpochs, nameString)

else:
    trained_model = keras.models.load_model('model.h5')

# PREDICTION
#
#
#

MFC_test = load_images_from_folder('mfcc/%s_test_l'%(nameString))

X_test = np.asarray(MFC_test)
X_test = np.array([x.reshape( (40, 33, 1) ) for x in X_test])
y_test = np.ones(200)

trained_model = keras.models.load_model('model.h5')
prediction = trained_model.predict(X_test)
prediction = np.asarray(prediction)
hangover_prediction = hangover_highlights(prediction)

print(hangover_prediction)


#getBaselinePrediction(audMonoGame, sample_rate)
