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
        return func(fp.numpy().decode())
    return _py_loader_function


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Benchmark audio loading in tensorflow'
    )
    parser.add_argument('--ext', type=str, default="wav")
    args = parser.parse_args()
    repeat = 3
    columns = [
        'ext',
        'lib',
        'duration',
        'time',
    ]

    store = utils.DF_writer(columns)

    libs = [
        'stempeg',
        'soxbindings',
        'ar_ffmpeg',
        'aubio',
        'pydub',
        'soundfile',
        'librosa',
        'scipy',
        'scipy_mmap',
        'tf_decode_wav',
        'tfio_fromaudio'
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                append = False
                duration = int(audio_dir)
                audio_files = get_files(
                    dir=os.path.join(root, audio_dir),
                    extension=args.ext
                )

                dataset = tf.data.Dataset.from_tensor_slices(audio_files)
                if lib in ["tf_decode_wav"]:
                    dataset = dataset.map(
                        lambda x: loaders.load_tf_decode_wav(x),
                        num_parallel_calls=1
                    )
                elif lib in ["tfio_fromaudio"]:
                    dataset = dataset.map(
                        lambda x: loaders.load_tfio_fromaudio(x),
                        num_parallel_calls=1
                    )
                elif lib in ["tfio_fromffmpeg"]:
                    dataset = dataset.map(
                        lambda x: loaders.load_tfio_fromffmpeg(x),
                        num_parallel_calls=1
                    )
                else:
                    loader_function = getattr(loaders, 'load_' + lib)
                    dataset = dataset.map(
                        lambda filename: tf.py_function(
                            _make_py_loader_function(loader_function),
                            [filename],
                            [tf.float32]
                        ),
                        num_parallel_calls=1
                    )

                dataset = dataset.apply(tf.data.experimental.ignore_errors())

                start = time.time()

                for i in range(repeat):
                    for audio in dataset:
                        value = tf.reduce_max(audio)
                        if value > 0.0:
                            append = True

                end = time.time()
                
                if append:
                    store.append(
                        ext=args.ext,
                        lib=lib,
                        duration=duration,
                        time=float(end-start) / (len(audio_files) * repeat),
                    )

    store.df.to_pickle("results/benchmark_%s_%s.pickle" % ("tf", args.ext))
