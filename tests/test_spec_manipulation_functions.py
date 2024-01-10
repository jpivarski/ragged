# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.broadcast_arrays is not None
    assert ragged.broadcast_to is not None
    assert ragged.concat is not None
    assert ragged.expand_dims is not None
    assert ragged.flip is not None
    assert ragged.permute_dims is not None
    assert ragged.reshape is not None
    assert ragged.roll is not None
    assert ragged.squeeze is not None
    assert ragged.stack is not None


@pytest.mark.parametrize("device", devices)
def test_broadcast_arrays(device, x, y):
    x_bc, y_bc = ragged.broadcast_arrays(x.to_device(device), y.to_device(device))
    if x.shape == () and y.shape == ():
        assert x_bc.shape == ()
        assert y_bc.shape == ()
    else:
        assert x_bc.shape == y_bc.shape
        if x_bc.shape == (3,):
            assert (x_bc * 0).tolist() == (y_bc * 0).tolist() == [0, 0, 0]
        if x_bc.shape == (3, None):
            assert (x_bc * 0).tolist() == (y_bc * 0).tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]


def test_concat(x, y):
    if x.ndim != y.ndim:
        with pytest.raises(ValueError, match="same number of dimensions"):
            ragged.concat([x, y])

    elif x.ndim == 0:
        with pytest.raises(ValueError, match="zero-dimensional"):
            ragged.concat([x, y])

    elif x.ndim == 1:
        assert ragged.concat([x, y], axis=None).tolist() == x.tolist() + y.tolist()
        assert ragged.concat([x, y], axis=0).tolist() == x.tolist() + y.tolist()

    else:
        assert ragged.concat([x, y], axis=None).tolist() == [
            1.1,
            2.2,
            3.3,
            4.4,
            5.5,
            1.1,
            2.2,
            3.3,
            4.4,
            5.5,
        ]
        assert ragged.concat([x, y], axis=0).tolist() == [  # type: ignore[comparison-overlap]
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert ragged.concat([x, y], axis=1).tolist() == [  # type: ignore[comparison-overlap]
            [1.1, 2.2, 3.3, 1.1, 2.2, 3.3],
            [],
            [4.4, 5.5, 4.4, 5.5],
        ]
