import frame_video_comparison
from frame_video_comparison import video_to_frames
import audio_processing

# Compares the highlights video to the full game footage
# Outputs a list of time stamps where similarities were found, i.e. game highlights
# video_path = 'video/...'
# frames_dir = 'video/frames/'
# overwrite = True/False

def get_highlight_list(gamePath, sumPath, framesPath, overwrite, every, chunk_size):
    # Extracts frames from the game summary video
    video_to_frames(video_path = sumPath, frames_dir = framesPath, overwrite = overwrite, every = every, chunk_size = chunk_size)
    frames_jump_comparison = 10 #compares every 10 images (i.e. 5 per seconds for 50fps)
    # Compares the extracted frames with the original game footage and when the similarity is almost 1, the frame number is stored in a list
    highlights_frame_number = frame_video_comparison.compare_frame_video(framesPath, gamePath, frames_jump_comparison)
    print(highlights_frame_number)
    return highlights_frame_number


# WIPPPP
# Summarizes the list of highlights
def process_highlights_list(highlights_frames_list):
    processed_list = highlights_frames_list
    return processed_list