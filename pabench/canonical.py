"""Canonical tensor conversion: anything -> float32 (channels, frames) torch tensor.

Every loader in the benchmark funnels its decoded output through `to_tensor` before
timing stops, so every library is timed to the same target representation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

Layout = Literal["channels_first", "frames_first"]

_VALID_LAYOUTS = ("channels_first", "frames_first")


def _to_numpy(data: object) -> np.ndarray:
    """Coerce numpy arrays, torch tensors, and `.numpy()`-exposing objects to ndarray."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    numpy_method = getattr(data, "numpy", None)
    if callable(numpy_method):
        return numpy_method()
    return np.asarray(data)


def to_tensor(data: object, layout: Layout) -> torch.Tensor:
    """Convert decoded audio to a contiguous float32 `(channels, frames)` tensor.

    `data` may be a numpy array, a torch tensor, or any object exposing `.numpy()`.
    1-D input is treated as single-channel audio and gains a leading channel axis.
    `layout` states how a 2-D input is arranged; it is ignored for 1-D input since
    there is only one channel either way.
    """
    if layout not in _VALID_LAYOUTS:
        raise ValueError(f"unknown layout: {layout!r} (expected one of {_VALID_LAYOUTS})")

    arr = _to_numpy(data)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # (1, frames): already channels-first
    elif arr.ndim == 2:
        if layout == "frames_first":
            arr = arr.T  # -> (channels, frames), not yet contiguous
    else:
        raise ValueError(f"to_tensor expects 1-D or 2-D data, got {arr.ndim}-D")

    # `np.ascontiguousarray` is a no-op (no copy) when `arr` is already a
    # contiguous float32 array, and copies only when the transpose above or a
    # dtype cast makes one necessary.
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(arr)
