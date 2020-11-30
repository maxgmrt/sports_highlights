import skimage
from skimage import io
import librosa
import numpy as np
import os
import csv
from numpy import genfromtxt
from skimage import img_as_ubyte

# outputs the one-channel version of a two-channels audio file
def getMono(audio_data):
    return audio_data.sum(axis=1) / 2

# returns the amplitude of a mono audio block of duration (t2-t1)
def getEnergyBlockTime(audio_data, t1, t2, rate):
    n1 = t1 * rate
    n2 = t2 * rate
    return np.sum(audio_data[n1:n2]**2)

# returns the amplitude of a mono audio block of duration (n2-n1)
def getEnergyBlockFrame(audio_data,n1,n2):
    return np.sum(audio_data[n1:n2]**2)


def getBaselinePrediction(audioFile):
    audMono, sample_rate = librosa.load(audioFile, sr=16000, mono=True)
    energy = []
    for i in range(int(len(audMono)/sample_rate)-1):
        energy.append(getEnergyBlockFrame(audMono, sample_rate*i, sample_rate*(i+1)))
        
    energy_array = np.asarray(energy)
    max_energy = max(energy_array)

    baseline = []
    for e in energy:
        if e > 0.9 * max_energy:
            baseline.append(1)
        else:
            baseline.append(0)
    return baseline


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

    if (test == True):

        nMFCinSec = int(10/multiplier)
        hop_length = int(multiplier * sample_rate / 32)
        for h in range(int(len(audMono)/sample_rate)):
            for i in range(nMFCinSec):
                if not os.path.exists('mfcc/test/%s/'%(nameString)):
                    os.makedirs('mfcc/test/%s/'%(nameString))
                out = 'mfcc/test/%s/MFC_%s_%s.png'%(nameString,h,i)
                # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
                start = int(h*sample_rate + i*(sample_rate/(10/multiplier)))
                # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
                stop = int(start + multiplier*sample_rate)

                write_mfcc_image(audMono[start:stop], sr=sample_rate, out=out, hop_length=hop_length, write=True)

    for h in highlights_timestamps:
        h = int(h)
        specs = []
        nMFCinSec = int(10/multiplier)
        hop_length = int(multiplier * sample_rate / 32)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s/%s%s_h/'%(text,prefix,nameString)):
                os.makedirs('mfcc/%s/%s%s_h/'%(text,prefix,nameString))
            out = 'mfcc/%s/%s%s_h/MFC_%s_%s.png'%(text,prefix,nameString,h,i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h*sample_rate + i*(sample_rate/(10/multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier*sample_rate)

            write_mfcc_image(audMono[start:stop], sr=sample_rate, out=out, hop_length=hop_length, write=True)

    for l in lowlights_timestamps:
        l = int(l)
        specs = []
        nMFCinSec = int(10 / multiplier)
        hop_length = int(multiplier * sample_rate / 32)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s/%s%s_l/' % (text, prefix, nameString)):
                os.makedirs('mfcc/%s/%s%s_l/' % (text, prefix, nameString))
            out = 'mfcc/%s/%s%s_l/MFC_%s_%s.png' % (text, prefix, nameString, l, i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h * sample_rate + i * (sample_rate / (10 / multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier * sample_rate)

            write_mfcc_image(audMono[start:stop], sr=sample_rate, out=out, hop_length=hop_length, write=True)


def generateTrainingMFCCs(audiosTrainPath, labelsPath, nameString):
    print("Generating training MFCCs...")
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

        audioFile = '%s/%s.mp3'%(audiosTrainPath, nameString)
        generateMFCC(audioFile, nameString, highlights_timestamps, lowlights_timestamps)
    print("Successfully generated training MFCCs!")