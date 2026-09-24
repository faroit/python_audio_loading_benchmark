import numpy as np
import pytest
import torch

from pabench.canonical import to_tensor


def test_channels_first_identity():
    data = np.arange(6, dtype=np.float64).reshape(2, 3)  # (channels, frames)
    tensor = to_tensor(data, "channels_first")
    assert tensor.shape == (2, 3)
    assert torch.equal(tensor, torch.tensor(data, dtype=torch.float32))


def test_frames_first_transposes():
    data = np.arange(6, dtype=np.float64).reshape(3, 2)  # (frames, channels)
    tensor = to_tensor(data, "frames_first")
    assert tensor.shape == (2, 3)
    expected = torch.tensor(data.T.copy(), dtype=torch.float32)
    assert torch.equal(tensor, expected)


def test_values_survive_transpose():
    # Distinct, non-symmetric per-channel data so a transpose bug (e.g. simply
    # reshaping instead of transposing) would be caught.
    data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)  # (frames=3, ch=2)
    tensor = to_tensor(data, "frames_first")
    assert tensor.shape == (2, 3)
    # channel 0 is column 0 of `data`: [0, 2, 4]; channel 1 is column 1: [1, 3, 5]
    assert torch.equal(tensor[0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(tensor[1], torch.tensor([1.0, 3.0, 5.0]))


def test_1d_promotion_channels_first():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = to_tensor(data, "channels_first")
    assert tensor.shape == (1, 3)
    assert torch.equal(tensor, torch.tensor([[1.0, 2.0, 3.0]]))


def test_1d_promotion_frames_first():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = to_tensor(data, "frames_first")
    assert tensor.shape == (1, 3)
    assert torch.equal(tensor, torch.tensor([[1.0, 2.0, 3.0]]))


def test_dtype_is_float32():
    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    tensor = to_tensor(data, "channels_first")
    assert tensor.dtype == torch.float32


def test_result_is_contiguous():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    tensor = to_tensor(data, "frames_first")
    assert tensor.is_contiguous()

    data2 = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor2 = to_tensor(data2, "channels_first")
    assert tensor2.is_contiguous()


def test_torch_tensor_input_accepted():
    data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    tensor = to_tensor(data, "channels_first")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (2, 3)


def test_object_exposing_numpy_method_accepted():
    class Wrapper:
        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = to_tensor(Wrapper(data), "channels_first")
    assert torch.equal(tensor, torch.tensor(data))


def test_unknown_layout_rejected():
    data = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        to_tensor(data, "not_a_layout")


def test_3d_rejected():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        to_tensor(data, "channels_first")
