import matplotlib
matplotlib.use('Agg')
import os
import os.path
import random
import time
import soundfile as sf
import argparse
import librosa
import utils
import loaders
import tensorflow as tf


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


def _make_py_loader_function(func):
    def _py_loader_function(fp):
        return func(fp.decode())
    return _py_loader_function


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark audio loading in tensorflow')
    parser.add_argument('--ext', type=str, default="wav")
    args = parser.parse_args()

    columns = [
        'ext',
        'lib',
        'duration',
        'time',
    ]

    store = utils.DF_writer(columns)

    libs = [
        'ar_gstreamer',
        'ar_ffmpeg',
        'ar_mad',
        'aubio',
        'pydub',
        'soundfile',
        'librosa',
        'scipy',
        'scipy_mmap',
        'tf_decode'
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                try:
                    duration = int(audio_dir)
                    audio_files = get_files(dir=os.path.join(root, audio_dir), extension=args.ext)

                    dataset = tf.data.Dataset.from_tensor_slices(audio_files)
                    if lib == "tf_decode":
                        dataset = dataset.map(lambda x: loaders.load_tf_decode(x, args.ext))
                    else:
                        loader_function = getattr(loaders, 'load_' + lib)
                        dataset = dataset.map(
                            lambda filename: tf.py_func(
                                _make_py_loader_function(loader_function), 
                                [filename], 
                                [tf.float32]
                            )
                        )

                    dataset = dataset.batch(1)
                    start = time.time()
                    iterator = dataset.make_one_shot_iterator()
                    next_audio = iterator.get_next()
                    with tf.Session() as sess:
                        for i in range(len(audio_files)):
                            try:
                                value = sess.run(tf.reduce_max(next_audio))
                            except tf.errors.OutOfRangeError:
                                break

                    end = time.time()
                    
                    store.append(
                        ext=args.ext,
                        lib=lib,
                        duration=duration,
                        time=float(end-start) / len(audio_files),
                    )
                except:
                    continue

    store.df.to_pickle("results/benchmark_%s_%s.pickle" % ("tf", args.ext))

