#!/bin/sh

# Script that creates a video from a lot of numbered images.
# You need to have installed 'ffmpeg'.
#
# First param:
# 	Folder with numbered jpg files (%07d.jpg).
# 	Default: current dir
# Second param:
# 	FPS.
#	Default: 4
# Third param:
#	Output filename. You can use any format supported by 'ffmpeg'.
#	Default: <FOLDER WITH IMAGES>/video.avi
# Fourth param:
#	Video format.
#	Default: Filename extension.

FOLDER=${1:-.}
FPS=${2:-4}
OUTPUT_FILE=${3:-$FOLDER/video.avi}
FORMAT=${4:-${OUTPUT_FILE##*.}}

ffmpeg -r $FPS -i $FOLDER/%07d.jpg -f $FORMAT -c:a mp3 -c:v libx264 $OUTPUT_FILE;
