import numpy as np

def hangover_highlights(prediction):
    average_prediction = []
    hangover_prediction = []

    for i in range(len(prediction)-10):
        average_prediction.append(np.mean(prediction[i:i+9]))
        
    for i in range(int(len(average_prediction)/5-5)):
        if ((average_prediction[5*i]>0.5)&
            (average_prediction[5*i+1]>0.5)&
            (average_prediction[5*i+2]>0.5)&
            (average_prediction[5*i+3]>0.5)&
            (average_prediction[5*i+4]>0.5)):
            
            hangover_prediction.append(1)
        
        else:
            hangover_prediction.append(0)
            
    return hangover_prediction


def processPrediction(prediction):
    maxPred = max(prediction)
    minPred = min(prediction)

    processedPred = []
    for i in prediction:
        if (maxPred - i) ^ 2 > (minPred - i) ^ 2:
            processedPred.append(0)
        else:
            processedPred.append(1)
    return processedPred