from scipy.io import wavfile
import audioread.rawread
import audioread.gstdec
import audioread.maddec
import audioread.ffdec
import matplotlib.pyplot as plt
import soundfile as sf
import aubio
from pydub import AudioSegment
import torchaudio
import numpy as np
import tensorflow as tf
import librosa

"""
Some of the code taken from: 
https://github.com/aubio/aubio/blob/master/python/demos/demo_reading_speed.py
"""


def load_tf_decode(fp, ext="wav", rate=44100):
    audio_binary = tf.read_file(fp)
    audio_decoded = tf.contrib.ffmpeg.decode_audio(
        audio_binary, 
        file_format=ext, 
        samples_per_second=rate, 
        channel_count=1
    )
    return tf.cast(audio_decoded, tf.float32)


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
    sig, rate = torchaudio.load(fp)
    return sig


def load_soundfile(fp):
    sig, rate = sf.read(fp)
    return sig


def load_scipy(fp):
    rate, sig = wavfile.read(fp)
    sig = sig.astype('float32') / 32767
    return sig


def load_scipy_mmap(fp):
    rate, sig = wavfile.read(fp, mmap=True)
    sig = sig.astype('float32') / 32767
    return sig


def load_ar_gstreamer(fp):
    with audioread.gstdec.GstAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = _convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_ar_mad(fp):
    with audioread.maddec.MadAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = _convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_ar_ffmpeg(fp):
    with audioread.ffdec.FFmpegAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = _convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_pydub(fp):
    song = AudioSegment.from_file(fp)
    sig = np.asarray(song.get_array_of_samples(), dtype='float32')
    sig = sig.reshape(song.channels, -1) / 32767.
    return sig


def load_librosa(fp):
    sig, rate = librosa.load(fp, sr=None)
    return sig


def _convert_buffer_to_float(buf, n_bytes=2, dtype=np.float32):
    # taken from librosa.util.utils
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    out = scale * np.frombuffer(buf, fmt).astype(dtype)
    return out
