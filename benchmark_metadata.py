import os
import os.path
import time
import argparse
import utils
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
        'stempeg',
        'torchaudio',
        'aubio',
        'soundfile',
        'sox',
        'audioread',
        'pedalboard',
    ]

    for lib in libs:
        print("Testing: %s" % lib)
        for root, dirs, fnames in sorted(os.walk('AUDIO')):
            for audio_dir in dirs:
                # torchaudio segfaults for MP4
                if lib in ['torchaudio', 'sox'] and args.ext == 'mp4':
                    continue
                if lib == 'soundfile' and args.ext in ['mp3', 'mp4']:
                    continue

                duration = int(audio_dir)
                dataset = AudioFolder(
                        os.path.join(root, audio_dir),
                        lib='info_' + lib,
                        extension=args.ext
                )

                start = time.time()

                for i in range(3):
                    for fp in dataset.audio_files:
                        info = dataset.loader_function(fp)
                        info['duration']

                end = time.time()
                store.append(
                    ext=args.ext,
                    lib=lib,
                    duration=duration,
                    time=float(end-start) / (len(dataset) * 3),
                )

    store.df.to_pickle('results/benchmark_metadata_{}.pickle'.format(args.ext))
