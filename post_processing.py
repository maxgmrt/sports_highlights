import numpy as np

def hangover_highlights(prediction, threshold):
    average_prediction = []
    hangover_prediction = []

    for i in range(len(prediction)-10):
        average_prediction.append(np.mean(prediction[i:i+9]))
        
    for i in range(int(len(average_prediction)/10-10)):
        if ((average_prediction[10 * i] > threshold) &
            (average_prediction[10 * i + 1] > threshold) &
            (average_prediction[10 * i + 2] > threshold) &
            (average_prediction[10 * i + 3] > threshold) &
            (average_prediction[10 * i + 4] > threshold) &
            (average_prediction[10 * i + 5] > threshold) &
            (average_prediction[10 * i + 6] > threshold) &
            (average_prediction[10 * i + 7] > threshold) &
            (average_prediction[10 * i + 8] > threshold) &
            (average_prediction[10 * i + 9] > threshold)
        ):
            
            hangover_prediction.append(1)
        
        else:
            hangover_prediction.append(0)
            
    return hangover_prediction


def processPrediction(prediction):
    maxPred = max(prediction)
    minPred = min(prediction)

    processedPred = []
    for i in prediction:
        if (maxPred - i)**2 > (minPred - i)**2:
            processedPred.append(0)
        else:
            processedPred.append(1)
    return processedPred