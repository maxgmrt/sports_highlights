import numpy as np
from utils import load_images_from_folder
from utils import load_pickle_from_folder
from audio_processing import generateMFCC
from audio_processing import getVoxIsolatedMFC
import os


def generatePrediction(audioTestFile):
    # Generation of test MFCCs
    nameStringTest = audioTestFile.replace(".mp3", "").replace("audio/test/","")
    if not os.path.exists('mfcc/test/%s' % (nameStringTest)):
        os.makedirs('mfcc/test/%s' % (nameStringTest))
        generateMFCC(audioTestFile, nameStringTest,  [], [], macro=False, test=True)
    print('mfcc/test/%s' % nameStringTest)
    MFC_test = load_pickle_from_folder('mfcc/test/%s' % nameStringTest)
    MFC_test_shaped = []
    for m in MFC_test:
        if (m.shape[0] == 40) & (m.shape[1] == 32):
            MFC_test_shaped.append(m)
    X_test = np.asarray(MFC_test_shaped)
    X_test = np.array([x.reshape((40, 32, 1)) for x in X_test])

    means=[]

    for s in range(X_test.shape[0]):
        means.append(int(np.mean(X_test[s])))

    np.savetxt('MFCSpec_means_test.csv', means, delimiter=',')

    return X_test


def getWordFlowHighlights(audioTestFile):

    S_test, sample_rate = getVoxIsolatedMFC(audioTestFile)
#    if not os.path.exists('mfcc/test/%s_voxiso' % (nameStringTest)):
#        os.makedirs('mfcc/test/%s_voxiso' % (nameStringTest))
#        generateMFCC(audioTestFile, nameStringTest,  [], [], macro=True, test=True)
#
#    MFC_test_macro = load_images_from_folder('mfcc/test/MACRO_%s' % nameStringTest)
    amplitude_average = []
    for i in range(int(len(S_test))):
        amplitude_average.append(np.mean(S_test[i]))

    max_amp = max(amplitude_average)
    amplitude_average = np.asarray(amplitude_average)

    wordflow_pred = []
    for a in amplitude_average:
        if a > 0.8 * max_amp:
            wordflow_pred.append(1)
        else:
            wordflow_pred.append(0)
    wordflow_pred = np.asarray(wordflow_pred)
    return wordflow_pred


