from __future__ import print_function
import torch.utils
import os
import os.path
import torchaudio
import random
import time
import soundfile as sf
import argparse
from scipy.io import wavfile
import librosa
import numpy as np
import subprocess as sp
import os
DEVNULL = open(os.devnull, 'w')


def ffmpeg_load_audio(
    filename, sr=44100, mono=True, normalize=True, in_type=np.int16, out_type=np.float32
):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr


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


def read_audio(fp, lib="librosa"):
    if lib == "torchaudio":
        sig, _ = torchaudio.load(fp)
    elif lib == "soundfile":
        sig, _ = sf.read(fp)
        sig = torch.FloatTensor(sig).view(1, 1, -1)
    elif lib == "scipy":
        rate, sig = wavfile.read(fp)
        sig = torch.FloatTensor(sig).view(1, 1, -1)
    elif lib == "librosa":
        sig, sr = librosa.load(fp)
        sig = torch.FloatTensor(sig).view(1, 1, -1)
    elif lib == "ffmpeg_call":
        sig, sr = ffmpeg_load_audio(fp)
        sig = torch.FloatTensor(sig).view(1, 1, -1)
    return sig


class AudioFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        download=True,
        extension='wav',
        max_len=1024,
        lib="librosa",
    ):
        self.root = os.path.expanduser(root)
        self.data = []
        self.max_len = max_len
        self.audio_files = get_files(dir=self.root, extension=extension)
        self.lib = lib

    def __getitem__(self, index):
        audio = read_audio(random.choice(self.audio_files), lib=self.lib)
        return audio

    def __len__(self):
        return self.max_len


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ext', type=str, default="wav")
parser.add_argument('--lib', type=str, default="librosa")
parser.add_argument('--nsamples', type=int, default=1024)
args = parser.parse_args()

data = torch.utils.data.DataLoader(
    AudioFolder(
        'audio', max_len=args.nsamples, lib=args.lib, extension=args.ext
    ),
    batch_size=1
)

start = time.time()

for X in data:
    X.mean()

end = time.time()

print(end - start)
