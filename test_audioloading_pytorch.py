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


def audio_to_tensor(fp, lib="librosa"):
    if lib == "torchaudio":
        sig, _ = torchaudio.load(fp)
        return sig
    elif lib == "soundfile":
        sig, _ = sf.read(fp)
    elif lib == "scipy":
        rate, sig = wavfile.read(fp)
    elif lib == "librosa":
        sig, sr = librosa.load(fp)
    elif lib == "ffmpeg_call":
        sig, sr = ffmpeg_load_audio(fp)
    return torch.FloatTensor(sig).view(1, 1, -1)


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
        audio = audio_to_tensor(random.choice(self.audio_files), lib=self.lib)
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
