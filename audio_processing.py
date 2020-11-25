import skimage
from skimage import io
import librosa
import numpy as np
import os
import csv
from numpy import genfromtxt

# outputs the one-channel version of a two-channels audio file
def getMono(audio_data):
    return audio_data.sum(axis=1) / 2

# returns the amplitude of a mono audio block of duration (t2-t1)
def getEnergyBlockTime(audio_data,t1,t2):
    n1 = t1 * rate
    n2 = t2 * rate
    return np.sum(audio_data[n1:n2]**2)

# returns the amplitude of a mono audio block of duration (n2-n1)
def getEnergyBlockFrame(audio_data,n1,n2):
    return np.sum(audio_data[n1:n2]**2)


def getBaselinePrediction(audMono, sample_rate):
    energy = []
    for i in range(int(len(audMono)/sample_rate)-1):
        energy.append(getEnergyBlockFrame(audMono, sample_rate*i, sample_rate*(i+1)))
        
    energy = np.asarray(energy)

    highlights = np.where(energy>0.95*max(energy))
    lowlights = np.where(energy < 1.05 * min(energy))

    return highlights, lowlights


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def write_mfcc_image(y, sr, out, hop_length, write=False):
    # use log-melspectrogram
    # extract a fixed length window
    #start_sample = 0 # starting at beginning
    #length_samples = time_steps*hop_length
    #window = y[start_sample:start_sample+length_samples]
    mels = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=40)
    #mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    #img = 255-img # invert. make black==more energy
    if(write == True):
    # save as PNG
        skimage.io.imsave(out, mels)
    return mels


def generateMFCC(audioFile, nameString, highlights_timestamps, lowlights_timestamps, macro=False, test=False):
    #by default, it is set to generate training micro MFC spectrograms

    # LOAD MP3 FILE
    audMono, sample_rate = librosa.load(audioFile, sr=16000, mono=True)

    multiplier = 1
    text = 'train'
    prefix=''

    if (macro == True):
        multiplier = 5
        prefix = 'MACRO_'

    if (test):
        text = 'test'

    for h in highlights_timestamps:
        h = int(h)
        specs = []
        nMFCinSec = int(10/multiplier)
        hop_length = int(multiplier * sample_rate / 32)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s%s_%s_h/'%(prefix,nameString,text)):
                os.makedirs('mfcc/%s%s_%s_h/'%(prefix,nameString,text))
            out = 'mfcc/%s%s_%s_h/MFC_%s_%s.png'%(prefix,nameString,text,h,i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h*sample_rate + i*(sample_rate/(10/multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier*sample_rate)

            specs.append(write_mfcc_image(audMono[start:stop], sr=sample_rate, out=out, hop_length=hop_length, write=True))


    for l in lowlights_timestamps:
        specs = []
        nMFCinSec = int(10/multiplier)
        hop_length = int(multiplier * sample_rate / 32)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s%s_%s_l/'%(prefix,nameString,text)):
                os.makedirs('mfcc/%s%s_%s_l/'%(prefix,nameString,text))
            out = 'mfcc/%s%s_%s_l/MFC_%s_%s.png'%(prefix,nameString,text,h,i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h*sample_rate + i*(sample_rate/(10/multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier*sample_rate)

            specs.append(write_mfcc_image(audMono[start:stop], sr=sample_rate, out=out, hop_length=hop_length, write=True))


def generateTrainingMFCCs(audiosTrainPath, labelsPath, nameString):

    for filename in os.listdir(audiosTrainPath):
        #nameString = filename.replace(".mp3", "")  # get rid of the .mp3 extension
        #list = []

        highlights_timestamps = []
        lowlights_timestamps = []

        my_data = genfromtxt('%s/%s.csv'%(labelsPath, nameString), delimiter=',')
        for row in my_data:
            if row[0] > 0:
                highlights_timestamps.append(int(row[0]))
            if row[1] > 0:
                lowlights_timestamps.append(int(row[1]))
        print(lowlights_timestamps)
        print(lowlights_timestamps[10])
        audioFile = '%s/%s.mp3'%(audiosTrainPath, nameString)
        generateMFCC(audioFile, nameString, highlights_timestamps, lowlights_timestamps)