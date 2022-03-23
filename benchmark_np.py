import os
import os.path
import timeit
import argparse
import utils
import numpy as np
import functools
import loaders


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


class AudioFolder(object):
    def __init__(
        self,
        root,
        extension='wav'
    ):
        self.root = os.path.expanduser(root)
        self.data = []
        self.audio_files = get_files(dir=self.root, extension=extension)

    def __getitem__(self, index):
        return self.audio_files[index]

    def __len__(self):
        return len(self.audio_files)


def test_np_loading(fp, lib):
    load_function = getattr(loaders, 'load_' + lib)
    audio = load_function(fp)
    if np.max(audio) > 0:
        return True
    else:
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ext', type=str, default="wav")
    args = parser.parse_args()

    columns = [
        'ext',
        'lib',
        'duration',
        'time',
    ]

    store = utils.DF_writer(columns)

    # audio formats to be bench
    # libraries to be benchmarked
    libs = [
        'scipy',
        'stempeg',
        'soxbindings',
        'ar_ffmpeg',
        'aubio',
        'pydub',
        'soundfile',
        'librosa',
        'scipy_mmap',
        'pedalboard',
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                duration = int(audio_dir)
                dataset = AudioFolder(
                    os.path.join(root, audio_dir),
                    extension=args.ext
                )

                # for fp in dataset.audio_files:
                for fp in dataset.audio_files:
                    time = min(timeit.repeat(
                        functools.partial(test_np_loading, fp, lib),
                        number=3,
                        repeat=3
                    ))

                store.append(
                    ext=args.ext,
                    lib=lib,
                    duration=duration,
                    time=time,
                )

    store.df.to_pickle("results/benchmark_%s_%s.pickle" % ("np", args.ext))