import frame_video_comparison
from frame_video_comparison import video_to_frames

import csv
import cv2
from utils import frameToTimestamp

import audio_processing

# Compares the highlights video to the full game footage
# Outputs a list of time stamps where similarities were found, i.e. game highlights
# video_path = 'video/...'
# frames_dir = 'video/frames/'
# overwrite = True/False

def processGroundTruth(gameNames):
    overwrite = True
    for g in gameNames:
        print('Comparing summary and game footage of %s' % g)

        gameFile = 'video/%s.mp4' % g
        sumFile = 'video/summaries/%s_sum.mp4' % g
        framesPath = 'video/%s_frames' % (g)  # typically ../video/PSGvSCO_frames/

        cap = cv2.VideoCapture(gameFile)
        fpsGame = cap.get(cv2.CAP_PROP_FPS)
        print(fpsGame)

        cap = cv2.VideoCapture(sumFile)
        fpsSum = cap.get(cv2.CAP_PROP_FPS)
        print(fpsSum)

        # Every other 'every' frame is saved for later comparison
        every = fpsSum * 2
        chunk_size = 1000

        # Returns the frame numbers of the highlights in the original game footage
        highlights_frame_number = get_highlight_list(gameFile, fpsGame, sumFile, framesPath, overwrite, every, chunk_size)

        # and the timestamps
        highlights_timestamps = frameToTimestamp(highlights_frame_number, fpsGame)

        with open('labels/%s_GroundTruth.csv' % g, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(highlights_timestamps)


def get_highlight_list(gameFile, fpsGame, sumFile, framesPath, overwrite, every, chunk_size):
    # Extracts frames from the game summary video
    video_to_frames(video_path = sumFile, frames_dir = framesPath, overwrite = overwrite, every = every, chunk_size = chunk_size)
    frames_jump_comparison = fpsGame/5
    # Compares the extracted frames with the original game footage and when the similarity is almost 1, the frame number is stored in a list
    highlights_frame_number = frame_video_comparison.compare_frame_video(framesPath, gameFile, frames_jump_comparison)
    print(highlights_frame_number)
    return highlights_frame_number


# WIPPPP
# Summarizes the list of highlights
def process_highlights_list(highlights_frames_list):
    processed_list = highlights_frames_list
    return processed_list