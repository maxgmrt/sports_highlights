# sports_highlights

Detecting sports highlights using audio cues.

## Getting Started

### Prerequisites
All needed Python modules and their versions are in the requirements.txt file.


![Screenshot](folder_structure.jpg)

**For every file refering to a game, it is crucial that the name string (i.e. the name of the game) is the same for every file.**
Example: for **PSGvSCO** game, audio track is named 'PSGvSCO.mp3', video file is named 'PSGvSCO.mp4', labels file is called 'PSGvSCO.csv'


Training audio samples must be put under audio/train/.
Game footage must be put under video/. Summaries must be put under video/summaries/.
Labels referung to audio files must be put under labels/.

## Running the code
In a terminal, run:

    python3 main.py -t <0 or 1> -g <0 or 1> -m <0 or 1> -a <path to test audio> -v <path to test video>

mandatory arguments:
<-t>: defines whether or not training will be performed. Possible values: 1 (yes) or 0 (no)

<-g>: defines if algorithm must generate Ground-Truth from video comparison. If yes, a list of highlights will be stored in a .csv file in labels/groundtruth/. Possible values: 1 (yes) or 0 (no)

<-m>: defines if algorithm re-generates mfcc spectrograms of the training dataset. If yes, spectrograms are stored as .pkl files under mfcc/train/. Possible values: 1 (yes) or 0 (no)

<-a>: path to test audio

optional arguments:
<-v>: path to test video

## Author

Maximilien Gomart - maximilien.gomart@epfl.ch

## Acknowledgments

AudioVisual Communications Lab (LCAV) @ EPFL
Prof. Adam Scholefield
Eric Bezzam
