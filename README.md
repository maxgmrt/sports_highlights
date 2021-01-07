# sports_highlights

Detecting sports highlights using audio cues.

## Getting Started

### Prerequisites
All needed Python modules and their versions are in the requirements.txt file.


### File locations
![Screenshot](folder_structure.jpg)

**For every file refering to a game, it is crucial that the name string (i.e. the name of the game) is the same for every file.**
Example: for **PSGvSCO** game, audio track is named 'PSGvSCO.mp3', video file is named 'PSGvSCO.mp4', labels file is called 'PSGvSCO.csv'


Training audio samples must be put under audio/train/.
Game footage must be put under video/. Summaries must be put under video/summaries/.
Labels referung to audio files must be put under labels/.

