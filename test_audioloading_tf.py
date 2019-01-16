from __future__ import print_function
import os
import os.path
import random
import time
import soundfile as sf
import argparse
from scipy.io import wavfile
import librosa
import utils
import tensorflow as tf
tf.enable_eager_execution()


def get_files(dir, extension):
    audio_files = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(root, fname)
                item = path
                audio_files.append(item)
    return audio_files


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ext', type=str, default="wav")
parser.add_argument('--lib', type=str, default="librosa")
parser.add_argument('--nsamples', type=int, default=64)
args = parser.parse_args()

root = os.path.expanduser("audio")
base_audio_files = get_files(dir=root, extension=args.ext)

audio_files = [random.choice(base_audio_files) for x in range(args.nsamples)]

with tf.Session() as sess:
    dataset = tf.data.Dataset.from_tensor_slices(audio_files)

    def _parse_function(filename):
        audio_binary = tf.read_file(filename)
        audio_decoded = tf.contrib.ffmpeg.decode_audio(
            audio_binary, 
            file_format=args.ext, 
            samples_per_second=44100, 
            channel_count=1
        )
        audio = tf.cast(audio_decoded, tf.float32)
        return audio


    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)
    start = time.time()
    iterator = dataset.make_one_shot_iterator()

    next_audio = iterator.get_next()

    for i in range(args.nsamples):
        value = sess.run(tf.reduce_max(next_audio))

    end = time.time()


print(end - start)
