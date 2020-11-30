import numpy as np
from utils import load_images_from_folder
from audio_processing import generateMFCC
from audio_processing import getBaselinePrediction
from post_processing import hangover_highlights

def generatePrediction(audioTestFile, trained_model):

    # Generation of test MFCCs
    nameStringTest = audioTestFile.replace(".mp3", "").replace("audio/test/","")
    generateMFCC(audioTestFile, nameStringTest,  [], [], macro=False, test=True)
    MFC_test = load_images_from_folder('mfcc/test/%s' % nameStringTest)
    MFC_test_shaped = []
    for m in MFC_test:
        if (m.shape[0] == 40) & (m.shape[1] == 33):
            MFC_test_shaped.append(m)
    X_test = np.asarray(MFC_test_shaped)
    X_test = np.array([x.reshape( (40, 33, 1) ) for x in X_test])
    y_test = np.ones(len(MFC_test))

    # Prediction using trained model
    prediction = trained_model.predict(X_test)
    prediction_array = np.asarray(prediction)
    threshold = np.mean(prediction_array)
    hangover_prediction = hangover_highlights(prediction_array, threshold)
    model_prediction = np.asarray(hangover_prediction)

    # Prediction using baseline
    baseline = np.asarray(getBaselinePrediction(audioTestFile))

    # Load test reference
    my_data = np.genfromtxt('labels/%s.csv' % (nameStringTest), delimiter=',')
    test_reference = np.asarray(my_data)

    return model_prediction, baseline, test_reference