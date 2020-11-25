import os
import cv2


def load_images_from_folder(folder):
    spectrograms = []
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        imgbw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            spectrograms.append(imgbw)
    return spectrograms

# From a list of frame numbers, returns a list of time stamps
def frameToTimestamp(highlights_frame_number, fps_game):
    highlights_timestamps = []
    for i in highlights_frame_number:
        highlights_timestamps.append(i/fps_game)
    return highlights_timestamps