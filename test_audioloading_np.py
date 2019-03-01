import matplotlib
matplotlib.use('agg')
import os
import os.path
import random
import time
import argparse
import utils
import loaders
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
        download=True,
        extension='wav',
        lib="librosa",
    ):
        self.root = os.path.expanduser(root)
        self.data = []
        self.audio_files = get_files(dir=self.root, extension=extension)
        self.loader_function = getattr(loaders, lib)

    def __getitem__(self, index):
        return self.loader_function(self.audio_files[index])

    def __len__(self):
        return len(self.audio_files)


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
        'ar_gstreamer',
        'ar_ffmpeg',
        'ar_mad',
        'aubio',
        'pydub',
        'torchaudio', 
        'soundfile', 
        'librosa', 
        'scipy',
        'scipy_mmap'
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                try:
                    duration = int(audio_dir)
                    dataset = AudioFolder(
                            os.path.join(root, audio_dir), 
                            lib='load_' + lib,
                            extension=args.ext
                    )
                    
                    
                    start = time.time()

                    for fp in dataset.audio_files:
                        audio = dataset.loader_function(fp)
                        np.max(audio)

                    end = time.time()
                    store.append(
                        ext=args.ext,
                        lib=lib,
                        duration=duration,
                        time=float(end-start) / len(dataset),
                    )
                except:
                    continue


    sns.set_style("whitegrid")

    ordered_libs = store.df.time.groupby(
        store.df.lib
    ).mean().sort_values().index.tolist()

    plt.subplot()

    g = sns.catplot(
        x="duration", 
        y="time", 
        kind='point',
        hue_order=ordered_libs,
        hue='lib', 
        data=store.df,
        height=6.6, 
        aspect=1
    )

    plt.savefig('benchmark_pytorch.png')

    plt.subplot()
    sns.barplot(x="lib", y="time", data=store.df, order=ordered_libs)
    plt.savefig("barplot_pytorch{}.png")