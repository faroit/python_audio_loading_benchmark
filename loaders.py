from scipy.io import wavfile
import audioread.rawread
import audioread.ffdec
import matplotlib.pyplot as plt
import soundfile as sf
import aubio
from pydub import AudioSegment
import torchaudio
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import soxbindings as sox
import stempeg


"""
Some of the code taken from: 
https://github.com/aubio/aubio/blob/master/python/demos/demo_reading_speed.py
"""


@tf.function
def load_tfio_fromffmpeg(fp):
    # not supported yet
    audio = tfio.IOTensor.graph(tf.int16).from_ffmpeg(fp)
    return tf.cast(audio.to_tensor(), tf.float32) / 32767.0


@tf.function
def load_tfio_fromaudio(fp):
    audio = tfio.IOTensor.graph(tf.int16).from_audio(fp)
    return tf.cast(audio.to_tensor(), tf.float32) / 32767.0


@tf.function
def load_tf_decode_wav(fp, ext="wav", rate=44100):
    audio, rate = tf.audio.decode_wav(tf.io.read_file(fp))
    return tf.cast(audio, tf.float32)


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


def load_stempeg(fp):
    sig = stempeg.read._read_ffmpeg(
        fp,
        sample_rate=44100,
        channels=1,
        start=None,
        duration=None,
        dtype=np.float32,
        ffmpeg_format='f32le',
        stem_idx=0
    )
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


def load_ar_ffmpeg(fp):
    with audioread.ffdec.FFmpegAudioFile(fp) as f:
        total_frames = 0
        for buf in f:
            sig = _convert_buffer_to_float(buf)
            sig = sig.reshape(f.channels, -1)
            total_frames += sig.shape[1]
        return sig


def load_soxbindings(fp):
    tfm = sox.Transformer()
    array_out = tfm.build_array(input_filepath=fp)
    return array_out


def load_pydub(fp):
    song = AudioSegment.from_file(fp)
    sig = np.asarray(song.get_array_of_samples(), dtype='float32')
    sig = sig.reshape(song.channels, -1) / 32767.
    return sig


def load_librosa(fp):
    # loading with `sr=None` is disabling the internal resampling
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


def info_soundfile(fp):
    info = {}
    info['duration'] = sf.info(fp).duration
    info['samples'] = int(sf.info(fp).duration * sf.info(fp).samplerate)
    info['channels'] = sf.info(fp).channels
    info['sampling_rate'] = sf.info(fp).samplerate
    return info


def info_audioread(fp):
    info = {}
    with audioread.audio_open(fp) as f:
        info['duration'] = f.duration
    with audioread.audio_open(fp) as f:
        info['samples'] = int(f.duration * f.samplerate)
    with audioread.audio_open(fp) as f:
        info['channels'] = f.channels
    with audioread.audio_open(fp) as f:
        info['sampling_rate'] = f.samplerate
    return info


def info_aubio(fp):
    info = {}
    with aubio.source(fp) as f:
        info['duration'] = f.duration / f.samplerate
    with aubio.source(fp) as f:
        info['samples'] = f.duration
    with aubio.source(fp) as f:
        info['channels'] = f.channels
    with aubio.source(fp) as f:
        info['sampling_rate'] = f.samplerate
    return info


def info_sox(fp):
    info = {}
    info['duration'] = sox.file_info.duration(fp)
    info['samples'] = sox.file_info.num_samples(fp)
    info['channels'] = sox.file_info.channels(fp)
    info['sampling_rate'] = int(sox.file_info.sample_rate(fp))
    return info


def info_pydub(fp):
    info = {}
    f = AudioSegment.from_file(fp)
    info['duration'] = f.duration_seconds
    f = AudioSegment.from_file(fp)
    info['samples'] = int(f.frame_count())
    f = AudioSegment.from_file(fp)
    info['channels'] = f.channels
    f = AudioSegment.from_file(fp)
    info['sampling_rate'] = f.frame_rate
    return info


def info_torchaudio(fp):
    info = {}
    f, _ = torchaudio.info(fp)
    info['duration'] = f.length / f.rate
    f, _ = torchaudio.info(fp)
    info['samples'] = f.length
    f, _ = torchaudio.info(fp)
    info['channels'] = f.channels
    f, _ = torchaudio.info(fp)
    info['sampling_rate'] = f.rate
    return info
