# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

devices = ["cpu"]
ns = {"cpu": np}
try:
    import cupy as cp

    devices.append("cuda")
    ns["cuda"] = cp
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.arange is not None
    assert ragged.asarray is not None
    assert ragged.empty is not None
    assert ragged.empty_like is not None
    assert ragged.eye is not None
    assert ragged.from_dlpack is not None
    assert ragged.full is not None
    assert ragged.full_like is not None
    assert ragged.linspace is not None
    assert ragged.meshgrid is not None
    assert ragged.ones is not None
    assert ragged.ones_like is not None
    assert ragged.tril is not None
    assert ragged.triu is not None
    assert ragged.zeros is not None
    assert ragged.zeros_like is not None


@pytest.mark.parametrize("device", devices)
def test_arange(device):
    a = ragged.arange(5, 10, 2, device=device)
    assert a.tolist() == [5, 7, 9]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_empty(device):
    a = ragged.empty((2, 3, 5), device=device)
    assert a.shape == (2, 3, 5)
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_empty_ndim0(device):
    a = ragged.empty((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_empty_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.empty_like(a)
    assert (b * 0).tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_eye(device):
    a = ragged.eye(3, 5, k=1, device=device)
    assert a.tolist() == [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.skipif(
    not hasattr(np, "from_dlpack"), reason=f"np.from_dlpack not in {np.__version__}"
)
@pytest.mark.parametrize("device", devices)
def test_from_dlpack(device):
    a = ns[device].array([1, 2, 3, 4, 5])
    b = ragged.from_dlpack(a)
    assert b.tolist() == [1, 2, 3, 4, 5]
    assert isinstance(b._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_full(device):
    a = ragged.full(5, 3, device=device)
    assert a.tolist() == [3, 3, 3, 3, 3]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_full_ndim0(device):
    a = ragged.full((), 3, device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 3
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_full_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.full_like(a, 5)
    assert b.tolist() == [[5, 5, 5], [], [5, 5]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_linspace(device):
    a = ragged.linspace(5, 8, 5, device=device)
    assert a.tolist() == [5, 5.75, 6.5, 7.25, 8]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_ones(device):
    a = ragged.ones(5, device=device)
    assert a.tolist() == [1, 1, 1, 1, 1]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_ones_ndim0(device):
    a = ragged.ones((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 1
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_ones_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.ones_like(a)
    assert b.tolist() == [[1, 1, 1], [], [1, 1]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_zeros(device):
    a = ragged.zeros(5, device=device)
    assert a.tolist() == [0, 0, 0, 0, 0]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_zeros_ndim0(device):
    a = ragged.zeros((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 0
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_zeros_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.zeros_like(a)
    assert b.tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device
