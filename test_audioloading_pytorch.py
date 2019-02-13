import matplotlib
matplotlib.use('agg')
import torch.utils
import os
import os.path
import torchaudio
import torch.utils.data
import random
import time
import soundfile as sf
import argparse
from scipy.io import wavfile
import librosa
import utils
import seaborn as sns
from pydub import AudioSegment
import aubio
import numpy as np
import audioread.rawread
import audioread.gstdec
import audioread.maddec
import audioread.ffdec
import matplotlib.pyplot as plt


"""
Some of the code taken from: 
https://github.com/aubio/aubio/blob/master/python/demos/demo_reading_speed.py
"""

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


def load_aubio(fp):
    f = aubio.source(fp, hop_size=1024)
    sig = np.zeros(f.duration, dtype=aubio.float_type)
    total_frames = 0
    while True:
        samples, read = f()
        sig[total_frames:total_frames + read] = samples[:read]
        total_frames += read
        if read < f.hop_size:
            break
    return sig

def load_torchaudio(fp):
    sig, _ = torchaudio.load(fp)
    return sig

def load_soundfile(fp):
    sig, _ = sf.read(fp)
    return sig

def load_scipy(fp):
    _, sig = wavfile.read(fp)
    sig = sig.astype('float32') / 32767
    return sig

def load_scipy_mmap(fp):
    _, sig = wavfile.read(fp, mmap=True)
    sig = sig.astype('float32') / 32767
    return sig

def convert_buffer_to_float(buf, n_bytes=2, dtype=np.float32):
    # taken from librosa.util.utils
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    out = scale * np.frombuffer(buf, fmt).astype(dtype)
    return out

def load_audioread_gstreamer(fp):
    with audioread.gstdec.GstAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_audioread_mad(fp):
    with audioread.maddec.MadAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_audioread_ffmpeg(fp):
    with audioread.ffdec.FFmpegAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_pydub(fp):
    song = AudioSegment.from_file(fp)
    sig = np.asarray(song.get_array_of_samples(), dtype='float32')
    sig = sig.reshape(song.channels, -1) / 32767.
    return sig


def load_librosa(fp):
    sig, _ = librosa.load(fp, sr=None)
    return sig


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
        self.loader_function = globals()[lib]

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
        'audioread_gstreamer',
        'audioread_ffmpeg',
        'audioread_mad',
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
        for root, dirs, fnames in sorted(os.walk('audio')):
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
                        pass

                    end = time.time()

                    store.append(
                        ext=args.ext,
                        lib=lib,
                        duration=duration,
                        time=float(end-start) / len(data),
                    )
                except:
                    continue

    sns.set_style("whitegrid")

    print(store.df)
    g = sns.catplot(
        x="duration", 
        y="time", 
        kind='point', 
        hue='lib', 
        data=store.df,
        height=6.6, 
        aspect=1
    )

    plt.savefig('benchmark.png')
