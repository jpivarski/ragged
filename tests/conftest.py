# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import reprlib

import awkward as ak
import numpy as np
import pytest

import ragged


@pytest.fixture(scope="session", autouse=True)
def _patch_reprlib():
    if not hasattr(reprlib.Repr, "repr1_original"):

        def repr1(self, x, level):
            if isinstance(x, ragged.array):
                return self.repr_instance(x, level)
            return self.repr1_original(x, level)

        reprlib.Repr.repr1_original = reprlib.Repr.repr1  # type: ignore[attr-defined]
        reprlib.Repr.repr1 = repr1  # type: ignore[method-assign]


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x(request):
    if request.param == "regular":
        return ragged.array(np.array([1.0, 2.0, 3.0]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(10.0))


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x_lt1(request):
    if request.param == "regular":
        return ragged.array(np.array([0.1, 0.2, 0.3]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[0.1, 0.2, 0.3], [], [0.4, 0.5]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(0.5))


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x_bool(request):
    if request.param == "regular":
        return ragged.array(np.array([False, True, False]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[True, True, False], [], [False, False]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(True))


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x_int(request):
    if request.param == "regular":
        return ragged.array(np.array([0, 1, 2], dtype=np.int64))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[1, 2, 3], [], [4, 5]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(10, dtype=np.int64))


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x_complex(request):
    if request.param == "regular":
        return ragged.array(np.array([1 + 0.1j, 2 + 0.2j, 3 + 0.3j]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[1 + 0j, 2 + 0j, 3 + 0j], [], [4 + 0j, 5 + 0j]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(10 + 1j))


y = x
y_lt1 = x_lt1
y_bool = x_bool
y_int = x_int
y_complex = x_complex
