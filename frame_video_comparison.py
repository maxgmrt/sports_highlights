# Compare an image with every frame of a video to find the best match

import os
import operator
import time
import datetime
import warnings
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys
from sys import stdout
import zipp

from skimage import color
from skimage.metrics import structural_similarity as ssim


#record start time
start = time.clock()

#ignore non-contiguous skimage warning
warnings.filterwarnings("ignore", module="skimage")


def prepare_image(img):
    #open still image as rgb
    #img = cv2.imread(filename, cv2.IMREAD_COLOR)
    #shrink
    img = cv2.resize(img, (10, 10))
    #convert to b&w
    img = color.rgb2gray(img)
    return img


def best_match(similarities):
    d = max(similarities, key=lambda x:x['similarity'])
    best_frame_number = d['frame']
    best_similarity = d['similarity']
    return best_frame_number, best_similarity


def parse_video(images, video, frames_jump_comparison, verbose=False):
    #iterate through video frames
    
    #similarities = [{'frame': 0, 'similarity': 0}]
    count = 0
    
    #get current time
    fps_time = time.clock()
    
    cap = cv2.VideoCapture(video)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    highlights_frame_number = []  
    
    while(cap.isOpened()):
        ret, frame = cap.read()

        #break at EOF
        if (type(frame) == type(None)):
            break

        #increment frame counter

        print(count)
        print(total_frame)

        count += 1

        #resize current video frame
        small_frame = cv2.resize(frame, (10, 10))
        #convert to greyscale
        small_frame_bw = color.rgb2gray(small_frame)
        
        similarity = []
        #compare current frame to source image
        for i in range(len(images)):
            similarity.append(ssim(images[i], small_frame_bw))
           
            if(similarity[i] > 0.95):
                highlights_frame_number.append(count)

    cap.release()
    return highlights_frame_number


def sort_results(results, output=False):
    #sort results
    print('\n')
    sorted_results = sorted(results, key=operator.itemgetter('similarity'), reverse=True)
    n = 0
    print('\n--results:')
    for res in sorted_results:
        n += 1
        print('#%s\t%s\t%s\t: %s' % (n, res['filename'], res['frame'], res['similarity']))

        #save matched frames
        if output:
            save_frame(output, n, res['image'])


def save_frame(filename, n, image):
    fn, ext = filename.split('.')
    fn = '%s_%s.%s' % (fn, n, ext)
    cv2.imwrite(fn, image)


def walk(source_image, directory, number=1):
    results = []
    extentions = ['mp4', 'avi', 'mov', 'mkv', 'm4v']
    for root, dirs, files in os.walk(directory):
        for file in files:
            for ext in extentions:
                if file.endswith(ext):
                    video_fn = (os.path.join(root, file))
                    print(video_fn)
                    similarities = parse_video(source_image,
                                               video_fn,
                                               n_matches=number)
                     
                    #flatten results
                    for d in similarities:
                        results.append({'filename'   : video_fn,
                                        'frame'      : d['frame'],
                                        'similarity' : d['similarity'],
                                        'image'      : d['image']})

    return results


def compare_frame_video(source_image_path, video_path, frames_jump_comparison):
    source_images_raw = []

    if not os.path.exists(source_image_path):
        os.makedirs(source_image_path)

    for filename in os.listdir(source_image_path):
        source_images_raw.append(cv2.imread(os.path.join(source_image_path,filename)))
    source_images = []

    for i in range(len(source_images_raw)):
        source_images.append(prepare_image(source_images_raw[i]))
    print('\n--reading video:')
    highlights_frame_number = parse_video(source_images, video_path, frames_jump_comparison)

    return highlights_frame_number



def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=25):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, "{:010d}.jpg".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=25, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    # os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir)  # when done return the directory containing the frames