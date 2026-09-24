import pytest
import soundfile as sf

from pabench.corpus import (
    DEFAULT_SPECS,
    FORMATS,
    CorpusSpec,
    Fmt,
    corpus_files,
    ffmpeg_available,
    generate,
)


def test_formats_has_four_entries():
    assert len(FORMATS) == 4
    assert len(set(FORMATS)) == 4
    assert {f.key for f in FORMATS} == {"wav_pcm16", "wav_float", "flac_pcm16", "mp3"}


def test_fmt_key():
    assert Fmt("wav", "PCM_16").key == "wav_pcm16"
    assert Fmt("wav", "FLOAT").key == "wav_float"
    assert Fmt("flac", "PCM_16").key == "flac_pcm16"
    assert Fmt("mp3", "MP3").key == "mp3"


def test_default_specs_axis_values():
    assert len(DEFAULT_SPECS) == 32

    durations = {s.duration_s for s in DEFAULT_SPECS}
    channels = {s.channels for s in DEFAULT_SPECS}
    fmts = {s.fmt for s in DEFAULT_SPECS}

    assert durations == {1, 10, 60, 300}
    assert channels == {1, 2}
    assert fmts == set(FORMATS)
    assert all(s.sample_rate == 44100 for s in DEFAULT_SPECS)


def test_default_specs_are_unique_combinations():
    combos = {(s.duration_s, s.channels, s.fmt) for s in DEFAULT_SPECS}
    assert len(combos) == 32


def test_filenames_unique():
    filenames = [s.filename for s in DEFAULT_SPECS]
    assert len(filenames) == len(set(filenames))


def test_spec_frames():
    spec = CorpusSpec(duration_s=10, channels=2, fmt=Fmt("wav", "PCM_16"))
    assert spec.frames == 10 * 44100


def test_corpus_files_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="pabench gen"):
        corpus_files(tmp_path, specs=DEFAULT_SPECS[:1])


def _small_specs():
    return (
        CorpusSpec(duration_s=1, channels=1, fmt=Fmt("wav", "PCM_16")),
        CorpusSpec(duration_s=1, channels=1, fmt=Fmt("wav", "FLOAT")),
        CorpusSpec(duration_s=1, channels=2, fmt=Fmt("flac", "PCM_16")),
    )


def test_generate_writes_wav_and_flac_matching_requested_properties(tmp_path):
    specs = _small_specs()
    paths = generate(tmp_path, specs=specs, seed=0)

    assert len(paths) == len(specs)

    for spec, path in zip(specs, paths):
        assert path.exists()
        info = sf.info(str(path))
        assert info.samplerate == spec.sample_rate
        assert info.channels == spec.channels
        assert info.frames == spec.frames
        assert info.subtype == spec.fmt.subtype


def test_generate_peak_amplitude_within_half(tmp_path):
    specs = _small_specs()
    paths = generate(tmp_path, specs=specs, seed=0)

    for path in paths:
        data, _ = sf.read(str(path), dtype="float64")
        assert abs(data).max() <= 0.5 + 1e-6


def test_generate_is_deterministic_per_spec(tmp_path):
    specs = _small_specs()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate(dir_a, specs=specs, seed=0)
    generate(dir_b, specs=specs, seed=0)

    for spec in specs:
        path_a = dir_a / spec.filename
        path_b = dir_b / spec.filename
        assert path_a.read_bytes() == path_b.read_bytes()


def test_regenerating_single_file_matches_full_sweep(tmp_path):
    specs = _small_specs()

    full_dir = tmp_path / "full"
    generate(full_dir, specs=specs, seed=0)

    single_dir = tmp_path / "single"
    single_spec = specs[-1]
    generate(single_dir, specs=(single_spec,), seed=0)

    full_bytes = (full_dir / single_spec.filename).read_bytes()
    single_bytes = (single_dir / single_spec.filename).read_bytes()
    assert full_bytes == single_bytes


def test_corpus_files_returns_specs_and_paths(tmp_path):
    specs = _small_specs()
    generate(tmp_path, specs=specs, seed=0)

    result = corpus_files(tmp_path, specs=specs)
    assert len(result) == len(specs)
    for (spec, path), expected_spec in zip(result, specs):
        assert spec == expected_spec
        assert path == tmp_path / expected_spec.filename
        assert path.exists()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not available")
def test_generate_writes_nonempty_mp3(tmp_path):
    spec = CorpusSpec(duration_s=1, channels=2, fmt=Fmt("mp3", "MP3"))
    paths = generate(tmp_path, specs=(spec,), seed=0)

    assert len(paths) == 1
    path = paths[0]
    assert path.exists()
    assert path.stat().st_size > 0

    info = sf.info(str(path))
    assert info.samplerate == spec.sample_rate
    assert info.channels == spec.channels


def test_ffmpeg_available_returns_bool():
    assert isinstance(ffmpeg_available(), bool)
