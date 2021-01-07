import skimage
from skimage import io
import librosa
import numpy as np
import os
from numpy import genfromtxt
from skimage import img_as_ubyte
import pickle
import utils
from utils import scale_minmax

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
        if e > 0.8 * max_energy:
            baseline.append(1)
        else:
            baseline.append(0)
    return baseline


def getVoxIsolatedMFC(audioTestFile):
    audMono, sample_rate = librosa.load(audioTestFile, sr=16000, mono=True)

    S_test = []
    for i in range(int(len(audMono)/sample_rate-5)):
        S_full, phase = librosa.magphase(librosa.stft(audMono[i*sample_rate:sample_rate*(i+5)]))
        S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sample_rate)))


        S_filter = np.minimum(S_full, S_filter)

        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)


        S_vox = mask_v * S_full
        S_test.append(S_vox)

    #S_foreground is the spectrogram of the isolated speech
    return S_test, sample_rate




def write_mfcc_image(y, sr, out, hop_length, write=False):
    # use log-melspectrogram
    # extract a fixed length window
    #start_sample = 0 # starting at beginning
    #length_samples = time_steps*hop_length
    #window = y[start_sample:start_sample+length_samples]
    mels = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=40)
    #mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    #mels = librosa.util.normalize(mels)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255)
    #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    #img = 255-img # invert. make black==more energy
    img = np.uint8(img)
    if(write == True):
    # save as PNG
        skimage.io.imsave(out, img)
    return img


def renormalize(mels, min1, max1, min2, max2):
    delta1 = max1 - min1 #scalar
    delta2 = max2 - min2 #scalar
    return (delta2 * (mels - min1) / delta1) + min2


def write_mfcc(y, sr, out, write=False):
    mels = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #mels_scaled = scale_minmax(mels, 0, 1000)
    #mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    #mels = librosa.util.normalize(mels)
    # min-max scale to fit inside 8-bit range
    #norm_mels = renormalize(mels, mels.min(), mels.max(), 0, 1000)

    if(write == True):
    # save as PNG
        pickle.dump(mels, open('%s.pkl'%out, 'wb'))
    return


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
        for s in range(int(len(audMono)/sample_rate)):
            for i in range(nMFCinSec):
                if not os.path.exists('mfcc/test/%s/'%(nameString)):
                    os.makedirs('mfcc/test/%s/'%(nameString))
                out = 'mfcc/test/%s/MFC_%s_%s'%(nameString,str(s).zfill(5),i)
                # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
                start = int(s*sample_rate + i*(sample_rate/(10/multiplier)))
                # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
                stop = int(start + multiplier*sample_rate)
                # Generates 10 MFC Spectrograms per second on the whole audio file
                write_mfcc(audMono[start:stop], sr=sample_rate, out=out, write=True)


    for h in highlights_timestamps:
        print("Saving Highlight MFCC at timestamp t = %s s" % h)
        h = int(h)
        specs = []
        nMFCinSec = int(10/multiplier)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s/%s%s_h/'%(text,prefix,nameString)):
                os.makedirs('mfcc/%s/%s%s_h/'%(text,prefix,nameString))
            out = 'mfcc/%s/%s%s_h/MFC_%s_%s'%(text,prefix,nameString,str(h).zfill(5),i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h*sample_rate + i*(sample_rate/(10/multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier*sample_rate)

            write_mfcc(audMono[start:stop], sr=sample_rate, out=out, write=True)


    for l in lowlights_timestamps:
        print("Saving Lowlight MFCC at timestamp t = %s s" % l)
        l = int(l)
        specs = []
        nMFCinSec = int(10 / multiplier)

        for i in range(nMFCinSec):
            if not os.path.exists('mfcc/%s/%s%s_l/' % (text, prefix, nameString)):
                os.makedirs('mfcc/%s/%s%s_l/' % (text, prefix, nameString))
            out = 'mfcc/%s/%s%s_l/MFC_%s_%s' % (text, prefix, nameString, str(l).zfill(5), i)
            # MICRO: 1 MFC every 0.1 second / MACRO: 1 MFC every 0.5 second
            start = int(h * sample_rate + i * (sample_rate / (10 / multiplier)))
            # MICRO: Lasts for 1 seconds / MACRO: Lasts for 5 seconds
            stop = int(start + multiplier * sample_rate)

            write_mfcc(audMono[start:stop], sr=sample_rate, out=out, write=True)


def generateTrainingMFCCs(audiosTrainPath, labelsPath, gameName):
    print("Generating training MFCCs for game %s" % gameName)

    highlights_timestamps = []
    lowlights_timestamps = []

    my_data = genfromtxt('%s/%s.csv'%(labelsPath, gameName), delimiter=',')
    for row in my_data:
        if row[0] > 0:
            highlights_timestamps.append(int(row[0]))
        if row[1] > 0:
            lowlights_timestamps.append(int(row[1]))

    audioFile = '%s/%s.mp3'%(audiosTrainPath, gameName)
    generateMFCC(audioFile, gameName, highlights_timestamps, lowlights_timestamps)
    print("Successfully generated training MFCCs for game %s!" % gameName)