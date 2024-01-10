# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spec_array_object import _box, _unbox, array
from ._typing import Dtype

_type = type


def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    """
    Copies an array to a specified data type irrespective of type promotion rules.

    Args:
        x: Array to cast.
        dtype: Desired data type.
        copy: Ignored because `ragged.array` data buffers are immutable.

    Returns:
        An array having the specified data type. The returned array has the
        same `shape` as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.astype.html
    """

    copy  # noqa: B018, argument is ignored, pylint: disable=W0104

    return _box(type(x), *_unbox(x), dtype=dtype)


def can_cast(from_: Dtype | array, to: Dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according type
    promotion rules.

    Args:
        from: Input data type or array from which to cast.
        to: Desired data type.

    Returns:
        `True` if the cast can occur according to type promotion rules;
        otherwise, `False`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.can_cast.html
    """

    return bool(np.can_cast(from_, to))


@dataclass
class finfo_object:  # pylint: disable=C0103
    """
    Output of `ragged.finfo` with the following attributes.

    - bits (int): number of bits occupied by the real-valued floating-point
      data type.
    - eps (float): difference between 1.0 and the next smallest representable
      real-valued floating-point number larger than 1.0 according to the
      IEEE-754 standard.
    - max (float): largest representable real-valued number.
    - min (float): smallest representable real-valued number.
    - smallest_normal (float): smallest positive real-valued floating-point
      number with full precision.
    - dtype (np.dtype): real-valued floating-point data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.finfo.html
    """

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: np.dtype


def finfo(type: Dtype | array, /) -> finfo_object:  # pylint: disable=W0622
    """
    Machine limits for floating-point data types.

    Args:
        type: the kind of floating-point data-type about which to get
            information. If complex, the information is about its component
            data type.

    Returns:
        An object having the following attributes:

        - bits (int): number of bits occupied by the real-valued floating-point
          data type.
        - eps (float): difference between 1.0 and the next smallest
          representable real-valued floating-point number larger than 1.0
          according to the IEEE-754 standard.
        - max (float): largest representable real-valued number.
        - min (float): smallest representable real-valued number.
        - smallest_normal (float): smallest positive real-valued floating-point
          number with full precision.
        - dtype (np.dtype): real-valued floating-point data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.finfo.html
    """

    if not isinstance(type, np.dtype):
        if not isinstance(type, _type) and hasattr(type, "dtype"):
            out = np.finfo(type.dtype)
        else:
            out = np.finfo(np.dtype(type))
    else:
        out = np.finfo(type)
    return finfo_object(
        out.bits, out.eps, out.max, out.min, out.smallest_normal, out.dtype
    )


@dataclass
class iinfo_object:  # pylint: disable=C0103
    """
    Output of `ragged.iinfo` with the following attributes.

    - bits (int): number of bits occupied by the type.
    - max (int): largest representable number.
    - min (int): smallest representable number.
    - dtype (np.dtype): integer data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.iinfo.html
    """

    bits: int
    max: int
    min: int
    dtype: np.dtype


def iinfo(type: Dtype | array, /) -> iinfo_object:  # pylint: disable=W0622
    """
    Machine limits for integer data types.

    Args:
        type: The kind of integer data-type about which to get information.

    Returns:
        An object having the following attributes:

        - bits (int): number of bits occupied by the type.
        - max (int): largest representable number.
        - min (int): smallest representable number.
        - dtype (np.dtype): integer data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.iinfo.html
    """

    if not isinstance(type, np.dtype):
        if not isinstance(type, _type) and hasattr(type, "dtype"):
            out = np.iinfo(type.dtype)
        else:
            out = np.iinfo(np.dtype(type))
    else:
        out = np.iinfo(type)
    return iinfo_object(out.bits, out.max, out.min, out.dtype)


def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...]) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type "kind".

    Args:
        dtype: The input dtype.
        kind: Data type kind.
            If `kind` is a `dtype`, the function returns a boolean indicating
            whether the input `dtype` is equal to the dtype specified by `kind`.

            If `kind` is a string, the function returns a boolean indicating
            whether the input `dtype` is of a specified data type kind. The
            following dtype kinds must be supported:

                - `"bool"`: boolean data types (e.g., bool).
                - `"signed integer"`: signed integer data types (e.g., `int8`,
                  `int16`, `int32`, `int64`).
                - `"unsigned integer"`: unsigned integer data types (e.g.,
                  `uint8`, `uint16`, `uint32`, `uint64`).
                - `"integral"`: integer data types. Shorthand for
                  (`"signed integer"`, `"unsigned integer"`).
                - `"real floating"`: real-valued floating-point data types
                  (e.g., `float32`, `float64`).
                - `"complex floating"`: complex floating-point data types
                  (e.g., `complex64`, `complex128`).
                - `"numeric"`: numeric data types. Shorthand for (`"integral"`,
                  `"real floating"`, `"complex floating"`).

            If `kind` is a tuple, the tuple specifies a union of dtypes and/or
            kinds, and the function returns a boolean indicating whether the
            input `dtype` is either equal to a specified dtype or belongs to at
            least one specified data type kind.

    Returns:
        Boolean indicating whether a provided dtype is of a specified data type
        kind.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    """

    dtype  # noqa: B018, pylint: disable=W0104
    kind  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 54")  # noqa: EM101


def result_type(*arrays_and_dtypes: array | Dtype) -> Dtype:
    """
    Returns the dtype that results from applying the type promotion rules to
    the arguments.

    Args:
        arrays_and_dtypes: An arbitrary number of input arrays and/or dtypes.

    Returns:
        The dtype resulting from an operation involving the input arrays and dtypes.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.result_type.html
    """

    return np.result_type(*arrays_and_dtypes)
