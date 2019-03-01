import matplotlib
matplotlib.use('Agg')
import torch.utils
import os
import os.path
import random
import time
import argparse
import librosa
import utils
import loaders
import torch


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


class AudioFolder(torch.utils.data.Dataset):
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
        audio = self.loader_function(self.audio_files[index])
        return torch.FloatTensor(audio).view(1, 1, -1)

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
        'soundfile', 
        'librosa', 
        'scipy',
        'scipy_mmap'
        'torchaudio'
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                try:
                    duration = int(audio_dir)
                    data = torch.utils.data.DataLoader(
                        AudioFolder(
                            os.path.join(root, audio_dir), 
                            lib='load_' + lib,
                            extension=args.ext
                        ),
                        batch_size=1,
                        num_workers=0,
                        shuffle=False
                    )
                    start = time.time()

                    for X in data:
                        X.max()

                    end = time.time()
                    store.append(
                        ext=args.ext,
                        lib=lib,
                        duration=duration,
                        time=float(end-start) / len(data),
                    )
                except:
                    continue


    utils.plot_results(store.df, "np", args.ext)