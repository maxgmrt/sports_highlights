from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2

def edit_video(video_file, highlights, debug = False):

    cap = cv2.VideoCapture(video_file)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    game = VideoFileClip(video_file)
    clips = []
    print('%s highlights were detected' % highlights[0].shape[0])

    for f in range(highlights[0].shape[0]):
        target_name_file = 'clip%s.mp4' % f
        print(highlights[0][f])
        times_start = int(highlights[0][f]) - 3
        times_end = int(highlights[0][f]) + 3

        clips.append(game.subclip(times_start, times_end))

        if debug:
            clips[f].write_videofile(target_name_file)

    final_clip = concatenate_videoclips([clips[j] for j in range(highlights[0].shape[0])])
    final_clip.write_videofile('video/output/' + video_file.replace("/", "_"))
    return