#!/usr/bin/env bash

# Set the number of files to generate
NBFILES=10
DIR=AUDIO

for k in $(seq -f "%02g" 1 5 61); do
    mkdir $DIR/$k
    for i in $(seq 1 $NBFILES); do
        sox -n -r 44100 $DIR/$k/$i.wav synth "0:$k" whitenoise vol 0.5 fade q 1 "0:$k" 1
        ffmpeg -i $DIR/$k/$i.wav $DIR/$k/$i.m4a
        ffmpeg -i $DIR/$k/$i.wav $DIR/$k/$i.mp3
    done
done