import os
import cv2
import pickle

def load_images_from_folder(folder):
    spectrograms = []
    filenames = []

    for filename in os.listdir(folder):
        filenames.append(filename)
    filenames.sort()

    for f in filenames:
        img = cv2.imread(os.path.join(folder,f))
        imgbw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            spectrograms.append(imgbw)

    return spectrograms

def load_pickle_from_folder(folder):
    spectrograms = []
    filenames = []

    for filename in os.listdir(folder):
        filenames.append(filename)
    filenames.sort()

    for f in filenames:
        file = open('%s/%s' % (folder,f),"rb")
        pck = pickle.load(file)
        if pck is not None:
            spectrograms.append(pck)

    return spectrograms

# From a list of frame numbers, returns a list of time stamps
def frameToTimestamp(highlights_frame_number, fps_game):
    highlights_timestamps = []
    for i in highlights_frame_number:
        highlights_timestamps.append(i/fps_game)
    return highlights_timestamps