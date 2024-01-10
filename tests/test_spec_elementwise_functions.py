# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
"""

from __future__ import annotations

import warnings
from typing import Any

import awkward as ak
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy.array_api as xp

import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def first(x: ragged.array) -> Any:
    out = ak.flatten(x._impl, axis=None)[0] if x.shape != () else x._impl
    return xp.asarray(out.item(), dtype=x.dtype)


def test_existence():
    assert ragged.abs is not None
    assert ragged.acos is not None
    assert ragged.acosh is not None
    assert ragged.add is not None
    assert ragged.asin is not None
    assert ragged.asinh is not None
    assert ragged.atan is not None
    assert ragged.atan2 is not None
    assert ragged.atanh is not None
    assert ragged.bitwise_and is not None
    assert ragged.bitwise_invert is not None
    assert ragged.bitwise_left_shift is not None
    assert ragged.bitwise_or is not None
    assert ragged.bitwise_right_shift is not None
    assert ragged.bitwise_xor is not None
    assert ragged.ceil is not None
    assert ragged.conj is not None
    assert ragged.cos is not None
    assert ragged.cosh is not None
    assert ragged.divide is not None
    assert ragged.equal is not None
    assert ragged.exp is not None
    assert ragged.expm1 is not None
    assert ragged.floor is not None
    assert ragged.floor_divide is not None
    assert ragged.greater is not None
    assert ragged.greater_equal is not None
    assert ragged.imag is not None
    assert ragged.isfinite is not None
    assert ragged.isinf is not None
    assert ragged.isnan is not None
    assert ragged.less is not None
    assert ragged.less_equal is not None
    assert ragged.log is not None
    assert ragged.log1p is not None
    assert ragged.log2 is not None
    assert ragged.log10 is not None
    assert ragged.logaddexp is not None
    assert ragged.logical_and is not None
    assert ragged.logical_not is not None
    assert ragged.logical_or is not None
    assert ragged.logical_xor is not None
    assert ragged.multiply is not None
    assert ragged.negative is not None
    assert ragged.not_equal is not None
    assert ragged.positive is not None
    assert ragged.pow is not None
    assert ragged.real is not None
    assert ragged.remainder is not None
    assert ragged.round is not None
    assert ragged.sign is not None
    assert ragged.sin is not None
    assert ragged.sinh is not None
    assert ragged.square is not None
    assert ragged.sqrt is not None
    assert ragged.subtract is not None
    assert ragged.tan is not None
    assert ragged.tanh is not None
    assert ragged.trunc is not None


@pytest.mark.parametrize("device", devices)
def test_abs(device, x):
    result = ragged.abs(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.abs(first(x)) == first(result)
    assert xp.abs(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_abs_method(device, x):
    result = abs(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.abs(first(x)) == first(result)
    assert xp.abs(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_acos(device, x_lt1):
    result = ragged.acos(x_lt1.to_device(device))
    assert type(result) is type(x_lt1)
    assert result.shape == x_lt1.shape
    assert xp.acos(first(x_lt1)) == pytest.approx(first(result))
    assert xp.acos(first(x_lt1)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_acosh(device, x):
    result = ragged.acosh(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.acosh(first(x)) == pytest.approx(first(result))
    assert xp.acosh(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_add(device, x, y):
    result = ragged.add(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.add(first(x), first(y)) == first(result)
    assert xp.add(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_add_method(device, x, y):
    result = x.to_device(device) + y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.add(first(x), first(y)) == first(result)
    assert xp.add(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_add_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.add(first(x), first(y))
    x += y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_asin(device, x_lt1):
    result = ragged.asin(x_lt1.to_device(device))
    assert type(result) is type(x_lt1)
    assert result.shape == x_lt1.shape
    assert xp.asin(first(x_lt1)) == pytest.approx(first(result))
    assert xp.asin(first(x_lt1)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_asinh(device, x):
    result = ragged.asinh(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.asinh(first(x)) == pytest.approx(first(result))
    assert xp.asinh(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_atan(device, x):
    result = ragged.atan(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.atan(first(x)) == pytest.approx(first(result))
    assert xp.atan(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_atan2(device, x, y):
    result = ragged.atan2(y.to_device(device), x.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.atan2(first(y), first(x)) == pytest.approx(first(result))
    assert xp.atan2(first(y), first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_atanh(device, x_lt1):
    result = ragged.atanh(x_lt1.to_device(device))
    assert type(result) is type(x_lt1)
    assert result.shape == x_lt1.shape
    assert xp.atanh(first(x_lt1)) == pytest.approx(first(result))
    assert xp.atanh(first(x_lt1)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_and(device, x_int, y_int):
    result = ragged.bitwise_and(x_int.to_device(device), y_int.to_device(device))
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_and(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_and(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_and_method(device, x_int, y_int):
    result = x_int.to_device(device) & y_int.to_device(device)
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_and(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_and(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_and_inplace_method(device, x_int, y_int):
    x_int = x_int.to_device(device)
    y_int = y_int.to_device(device)
    z_int = xp.bitwise_and(first(x_int), first(y_int))
    x_int &= y_int
    assert first(x_int) == z_int
    assert x_int.dtype == z_int.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_invert(device, x_int):
    result = ragged.bitwise_invert(x_int.to_device(device))
    assert type(result) is type(x_int)
    assert result.shape == x_int.shape
    assert xp.bitwise_invert(first(x_int)) == first(result)
    assert xp.bitwise_invert(first(x_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_invert_method(device, x_int):
    result = ~x_int.to_device(device)
    assert type(result) is type(x_int)
    assert result.shape == x_int.shape
    assert xp.bitwise_invert(first(x_int)) == first(result)
    assert xp.bitwise_invert(first(x_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_left_shift(device, x_int, y_int):
    result = ragged.bitwise_left_shift(x_int.to_device(device), y_int.to_device(device))
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_left_shift(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_left_shift(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_left_shift_method(device, x_int, y_int):
    result = x_int.to_device(device) << y_int.to_device(device)
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_left_shift(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_left_shift(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_left_shift_inplace_method(device, x_int, y_int):
    x_int = x_int.to_device(device)
    y_int = y_int.to_device(device)
    z_int = xp.bitwise_left_shift(first(x_int), first(y_int))
    x_int <<= y_int
    assert first(x_int) == z_int
    assert x_int.dtype == z_int.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_or(device, x_int, y_int):
    result = ragged.bitwise_or(x_int.to_device(device), y_int.to_device(device))
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_or(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_or(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_or_method(device, x_int, y_int):
    result = x_int.to_device(device) | y_int.to_device(device)
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_or(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_or(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_or_inplace_method(device, x_int, y_int):
    x_int = x_int.to_device(device)
    y_int = y_int.to_device(device)
    z_int = xp.bitwise_or(first(x_int), first(y_int))
    x_int |= y_int
    assert first(x_int) == z_int
    assert x_int.dtype == z_int.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_right_shift(device, x_int, y_int):
    result = ragged.bitwise_right_shift(
        x_int.to_device(device), y_int.to_device(device)
    )
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_right_shift(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_right_shift(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_right_shift_method(device, x_int, y_int):
    result = x_int.to_device(device) >> y_int.to_device(device)
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_right_shift(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_right_shift(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_right_shift_inplace_method(device, x_int, y_int):
    x_int = x_int.to_device(device)
    y_int = y_int.to_device(device)
    z_int = xp.bitwise_right_shift(first(x_int), first(y_int))
    x_int >>= y_int
    assert first(x_int) == z_int
    assert x_int.dtype == z_int.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_xor(device, x_int, y_int):
    result = ragged.bitwise_xor(x_int.to_device(device), y_int.to_device(device))
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_xor(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_xor(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_xor_method(device, x_int, y_int):
    result = x_int.to_device(device) ^ y_int.to_device(device)
    assert type(result) is type(x_int) is type(y_int)
    assert result.shape in (x_int.shape, y_int.shape)
    assert xp.bitwise_xor(first(x_int), first(y_int)) == first(result)
    assert xp.bitwise_xor(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_bitwise_xor_inplace_method(device, x_int, y_int):
    x_int = x_int.to_device(device)
    y_int = y_int.to_device(device)
    z_int = xp.bitwise_xor(first(x_int), first(y_int))
    x_int ^= y_int
    assert first(x_int) == z_int
    assert x_int.dtype == z_int.dtype


@pytest.mark.parametrize("device", devices)
def test_ceil(device, x):
    result = ragged.ceil(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.ceil(first(x)) == first(result)
    assert xp.ceil(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_ceil_int(device, x_int):
    result = ragged.ceil(x_int.to_device(device))
    assert type(result) is type(x_int)
    assert result.shape == x_int.shape
    assert xp.ceil(first(x_int)) == first(result)
    assert xp.ceil(first(x_int)).dtype == result.dtype


@pytest.mark.skipif(
    np.dtype("complex128") not in xp._dtypes._all_dtypes,
    reason=f"complex not allowed in np.array_api version {np.__version__}",
)
@pytest.mark.parametrize("device", devices)
def test_conj(device, x_complex):
    result = ragged.conj(x_complex.to_device(device))
    assert type(result) is type(x_complex)
    assert result.shape == x_complex.shape
    assert xp.conj(first(x_complex)) == first(result)
    assert xp.conj(first(x_complex)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_cos(device, x):
    result = ragged.cos(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.cos(first(x)) == pytest.approx(first(result))
    assert xp.cos(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_cosh(device, x):
    result = ragged.cosh(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.cosh(first(x)) == pytest.approx(first(result))
    assert xp.cosh(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_divide(device, x, y):
    result = ragged.divide(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.divide(first(x), first(y)) == first(result)
    assert xp.divide(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_divide_method(device, x, y):
    result = x.to_device(device) / y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.divide(first(x), first(y)) == first(result)
    assert xp.divide(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_divide_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.divide(first(x), first(y))
    x /= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_equal(device, x, y):
    result = ragged.equal(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.equal(first(x), first(y)) == first(result)
    assert xp.equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_equal_method(device, x, y):
    result = x.to_device(device) == y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.equal(first(x), first(y)) == first(result)
    assert xp.equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_exp(device, x):
    result = ragged.exp(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.exp(first(x)) == pytest.approx(first(result))
    assert xp.exp(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_expm1(device, x):
    result = ragged.expm1(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.expm1(first(x)) == pytest.approx(first(result))
    assert xp.expm1(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_floor(device, x):
    result = ragged.floor(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.floor(first(x)) == first(result)
    assert xp.floor(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_floor_int(device, x_int):
    result = ragged.floor(x_int.to_device(device))
    assert type(result) is type(x_int)
    assert result.shape == x_int.shape
    assert xp.floor(first(x_int)) == first(result)
    assert xp.floor(first(x_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_floor_divide(device, x, y):
    result = ragged.floor_divide(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.floor_divide(first(x), first(y)) == first(result)
    assert xp.floor_divide(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_floor_divide_method(device, x, y):
    result = x.to_device(device) // y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.floor_divide(first(x), first(y)) == first(result)
    assert xp.floor_divide(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_floor_divide_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.floor_divide(first(x), first(y))
    x //= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_floor_divide_int(device, x_int, y_int):
    with np.errstate(divide="ignore"):
        result = ragged.floor_divide(x_int.to_device(device), y_int.to_device(device))
        assert type(result) is type(x_int) is type(y_int)
        assert result.shape in (x_int.shape, y_int.shape)
        assert xp.floor_divide(first(x_int), first(y_int)) == first(result)
        assert xp.floor_divide(first(x_int), first(y_int)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_greater(device, x, y):
    result = ragged.greater(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.greater(first(x), first(y)) == first(result)
    assert xp.greater(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_greater_method(device, x, y):
    result = x.to_device(device) > y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.greater(first(x), first(y)) == first(result)
    assert xp.greater(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_greater_equal(device, x, y):
    result = ragged.greater_equal(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.greater_equal(first(x), first(y)) == first(result)
    assert xp.greater_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_greater_equal_method(device, x, y):
    result = x.to_device(device) >= y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.greater_equal(first(x), first(y)) == first(result)
    assert xp.greater_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.skipif(
    np.dtype("complex128") not in xp._dtypes._all_dtypes,
    reason=f"complex not allowed in np.array_api version {np.__version__}",
)
@pytest.mark.parametrize("device", devices)
def test_imag(device, x_complex):
    result = ragged.imag(x_complex.to_device(device))
    assert type(result) is type(x_complex)
    assert result.shape == x_complex.shape
    assert xp.imag(first(x_complex)) == first(result)
    assert xp.imag(first(x_complex)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_isfinite(device, x):
    result = ragged.isfinite(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.isfinite(first(x)) == first(result)
    assert xp.isfinite(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_isinf(device, x):
    result = ragged.isinf(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.isinf(first(x)) == first(result)
    assert xp.isinf(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_isnan(device, x):
    result = ragged.isnan(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.isnan(first(x)) == first(result)
    assert xp.isnan(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_less(device, x, y):
    result = ragged.less(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.less(first(x), first(y)) == first(result)
    assert xp.less(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_less_method(device, x, y):
    result = x.to_device(device) < y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.less(first(x), first(y)) == first(result)
    assert xp.less(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_less_equal(device, x, y):
    result = ragged.less_equal(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.less_equal(first(x), first(y)) == first(result)
    assert xp.less_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_less_equal_method(device, x, y):
    result = x.to_device(device) <= y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.less_equal(first(x), first(y)) == first(result)
    assert xp.less_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_log(device, x):
    result = ragged.log(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.log(first(x)) == pytest.approx(first(result))
    assert xp.log(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_log1p(device, x):
    result = ragged.log1p(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.log1p(first(x)) == pytest.approx(first(result))
    assert xp.log1p(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_log2(device, x):
    result = ragged.log2(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.log2(first(x)) == pytest.approx(first(result))
    assert xp.log2(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_log10(device, x):
    result = ragged.log10(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.log10(first(x)) == pytest.approx(first(result))
    assert xp.log10(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_logaddexp(device, x, y):
    result = ragged.logaddexp(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.logaddexp(first(x), first(y)) == pytest.approx(first(result))
    assert xp.logaddexp(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_logical_and(device, x_bool, y_bool):
    result = ragged.logical_and(x_bool.to_device(device), y_bool.to_device(device))
    assert type(result) is type(x_bool) is type(y_bool)
    assert result.shape in (x_bool.shape, y_bool.shape)
    assert xp.logical_and(first(x_bool), first(y_bool)) == first(result)
    assert xp.logical_and(first(x_bool), first(y_bool)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_logical_not(device, x_bool):
    result = ragged.logical_not(x_bool.to_device(device))
    assert type(result) is type(x_bool)
    assert result.shape == x_bool.shape
    assert xp.logical_not(first(x_bool)) == first(result)
    assert xp.logical_not(first(x_bool)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_logical_or(device, x_bool, y_bool):
    result = ragged.logical_or(x_bool.to_device(device), y_bool.to_device(device))
    assert type(result) is type(x_bool) is type(y_bool)
    assert result.shape in (x_bool.shape, y_bool.shape)
    assert xp.logical_or(first(x_bool), first(y_bool)) == first(result)
    assert xp.logical_or(first(x_bool), first(y_bool)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_logical_xor(device, x_bool, y_bool):
    result = ragged.logical_xor(x_bool.to_device(device), y_bool.to_device(device))
    assert type(result) is type(x_bool) is type(y_bool)
    assert result.shape in (x_bool.shape, y_bool.shape)
    assert xp.logical_xor(first(x_bool), first(y_bool)) == first(result)
    assert xp.logical_xor(first(x_bool), first(y_bool)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_multiply(device, x, y):
    result = ragged.multiply(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.multiply(first(x), first(y)) == first(result)
    assert xp.multiply(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_multiply_method(device, x, y):
    result = x.to_device(device) * y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.multiply(first(x), first(y)) == first(result)
    assert xp.multiply(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_multiply_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.multiply(first(x), first(y))
    x *= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_negative(device, x):
    result = ragged.negative(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.negative(first(x)) == pytest.approx(first(result))
    assert xp.negative(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_negative_method(device, x):
    result = -x.to_device(device)
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.negative(first(x)) == pytest.approx(first(result))
    assert xp.negative(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_not_equal(device, x, y):
    result = ragged.not_equal(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.not_equal(first(x), first(y)) == first(result)
    assert xp.not_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_not_equal_method(device, x, y):
    result = x.to_device(device) != y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.not_equal(first(x), first(y)) == first(result)
    assert xp.not_equal(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_positive(device, x):
    result = ragged.positive(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.positive(first(x)) == pytest.approx(first(result))
    assert xp.positive(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_positive_method(device, x):
    result = +x.to_device(device)
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.positive(first(x)) == pytest.approx(first(result))
    assert xp.positive(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_pow(device, x, y):
    result = ragged.pow(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.pow(first(x), first(y)) == first(result)
    assert xp.pow(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_pow_method(device, x, y):
    result = x.to_device(device) ** y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.pow(first(x), first(y)) == first(result)
    assert xp.pow(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_pow_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.pow(first(x), first(y))
    x **= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.skipif(
    np.dtype("complex128") not in xp._dtypes._all_dtypes,
    reason=f"complex not allowed in np.array_api version {np.__version__}",
)
@pytest.mark.parametrize("device", devices)
def test_real(device, x_complex):
    result = ragged.real(x_complex.to_device(device))
    assert type(result) is type(x_complex)
    assert result.shape == x_complex.shape
    assert xp.real(first(x_complex)) == first(result)
    assert xp.real(first(x_complex)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_remainder(device, x, y):
    result = ragged.remainder(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.remainder(first(x), first(y)) == first(result)
    assert xp.remainder(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_remainder_method(device, x, y):
    result = x.to_device(device) % y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.remainder(first(x), first(y)) == first(result)
    assert xp.remainder(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_remainder_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.remainder(first(x), first(y))
    x %= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_round(device, x):
    result = ragged.round(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.round(first(x)) == first(result)
    assert xp.round(first(x)).dtype == result.dtype


@pytest.mark.skipif(
    np.dtype("complex128") not in xp._dtypes._all_dtypes,
    reason=f"complex not allowed in np.array_api version {np.__version__}",
)
@pytest.mark.parametrize("device", devices)
def test_round_complex(device, x_complex):
    result = ragged.round(x_complex.to_device(device))
    assert type(result) is type(x_complex)
    assert result.shape == x_complex.shape
    assert xp.round(first(x_complex)) == first(result)
    assert xp.round(first(x_complex)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_sign(device, x):
    result = ragged.sign(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.sign(first(x)) == first(result)
    assert xp.sign(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_sin(device, x):
    result = ragged.sin(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.sin(first(x)) == pytest.approx(first(result))
    assert xp.sin(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_sinh(device, x):
    result = ragged.sinh(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.sinh(first(x)) == pytest.approx(first(result))
    assert xp.sinh(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_square(device, x):
    result = ragged.square(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.square(first(x)) == first(result)
    assert xp.square(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_sqrt(device, x):
    result = ragged.sqrt(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.sqrt(first(x)) == pytest.approx(first(result))
    assert xp.sqrt(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_subtract(device, x, y):
    result = ragged.subtract(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.subtract(first(x), first(y)) == first(result)
    assert xp.subtract(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_subtract_inplace_method(device, x, y):
    x = x.to_device(device)
    y = y.to_device(device)
    z = xp.subtract(first(x), first(y))
    x -= y
    assert first(x) == z
    assert x.dtype == z.dtype


@pytest.mark.parametrize("device", devices)
def test_subtract_method(device, x, y):
    result = x.to_device(device) - y.to_device(device)
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert xp.subtract(first(x), first(y)) == first(result)
    assert xp.subtract(first(x), first(y)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_tan(device, x):
    result = ragged.tan(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.tan(first(x)) == pytest.approx(first(result))
    assert xp.tan(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_tanh(device, x):
    result = ragged.tanh(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.tanh(first(x)) == pytest.approx(first(result))
    assert xp.tanh(first(x)).dtype == result.dtype


@pytest.mark.parametrize("device", devices)
def test_trunc(device, x):
    result = ragged.trunc(x.to_device(device))
    assert type(result) is type(x)
    assert result.shape == x.shape
    assert xp.trunc(first(x)) == first(result)
    assert xp.trunc(first(x)).dtype == result.dtype
