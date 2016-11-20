#!/bin/sh

# Creates heavy gifs (~ 3MB per minute of video).
#
# First param:
#	Video file to convert.

# Generate palette.
ffmpeg -i $1 -vf palettegen $1.png

# Generate gif.
ffmpeg -i $1 -pix_fmt rgb24 -i $1.png -lavfi "fps=4,scale=320:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y $1.gif
