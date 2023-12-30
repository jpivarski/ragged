# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/array_object.html
"""

from __future__ import annotations

import enum
import numbers
from typing import TYPE_CHECKING, Any, Union

import awkward as ak
import numpy as np
from awkward.contents import (
    Content,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)

from . import _import
from ._typing import (
    Device,
    Dtype,
    NestedSequence,
    PyCapsule,
    Shape,
    SupportsBufferProtocol,
    SupportsDLPack,
    numeric_types,
)


def _shape_dtype(layout: Content) -> tuple[Shape, Dtype]:
    node = layout
    shape: Shape = (len(layout),)
    while isinstance(node, (ListArray, ListOffsetArray, RegularArray)):
        if isinstance(node, RegularArray):
            shape = (*shape, node.size)
        else:
            shape = (*shape, None)
        node = node.content

    if isinstance(node, NumpyArray):
        shape = shape + node.data.shape[1:]
        return shape, node.data.dtype

    msg = f"Awkward Array type must have regular and irregular lists only, not {layout.form.type!s}"
    raise TypeError(msg)


# https://github.com/python/typing/issues/684#issuecomment-548203158
if TYPE_CHECKING:
    from enum import Enum

    class ellipsis(Enum):  # pylint: disable=C0103
        Ellipsis = "..."  # pylint: disable=C0103

    Ellipsis = ellipsis.Ellipsis  # pylint: disable=W0622

else:
    ellipsis = type(...)  # pylint: disable=C0103

GetSliceKey = Union[
    int,
    slice,
    ellipsis,
    None,
    tuple[Union[int, slice, ellipsis, None], ...],
    "array",
]

SetSliceKey = Union[
    int, slice, ellipsis, tuple[Union[int, slice, ellipsis], ...], "array"
]


class array:  # pylint: disable=C0103
    """
    Ragged array class and constructor.

    https://data-apis.org/array-api/latest/API_specification/array_object.html
    """

    # Constructors, internal functions, and other methods that are unbound by
    # the Array API specification.

    _impl: ak.Array | SupportsDLPack  # ndim > 0 ak.Array or ndim == 0 NumPy or CuPy
    _shape: Shape
    _dtype: Dtype
    _device: Device

    @classmethod
    def _new(cls, impl: ak.Array, shape: Shape, dtype: Dtype, device: Device) -> array:
        """
        Simple/fast array constructor for internal code.
        """

        out = cls.__new__(cls)
        out._impl = impl
        out._shape = shape
        out._dtype = dtype
        out._device = device
        return out

    def __init__(
        self,
        obj: (
            array
            | ak.Array
            | bool
            | int
            | float
            | complex
            | NestedSequence[bool | int | float | complex]
            | SupportsBufferProtocol
            | SupportsDLPack
        ),
        dtype: None | Dtype | type | str = None,
        device: None | Device = None,
        copy: None | bool = None,
    ):
        """
        Primary array constructor, same as `ragged.asarray`.

        Args:
            obj: Object to be converted to an array. May be a Python scalar, a
                (possibly nested) sequence of Python scalars, or an object
                supporting the Python buffer protocol or DLPack.
            dtype: Output array data type. If `dtype` is `None`, the output
                array data type is inferred from the data type(s) in `obj`.
                If all input values are Python scalars, then, in order of
                precedence,
                    - if all values are of type `bool`, the output data type is
                      `bool`.
                    - if all values are of type `int` or are a mixture of `bool`
                      and `int`, the output data type is `np.int64`.
                    - if one or more values are `complex` numbers, the output
                      data type is `np.complex128`.
                    - if one or more values are `float`s, the output data type
                      is `np.float64`.
            device: Device on which to place the created array. If device is
                `None` and `obj` is an array, the output array device is
                inferred from `obj`. If `"cpu"`, the array is backed by NumPy
                and resides in main memory; if `"cuda"`, the array is backed by
                CuPy and resides in CUDA global memory.
            copy: Boolean indicating whether or not to copy the input. If `True`,
                this function always copies. If `False`, the function never
                copies for input which supports the buffer protocol and raises
                a ValueError in case a copy would be necessary. If `None`, the
                function reuses the existing memory buffer if possible and
                copies otherwise.
        """

        if isinstance(obj, array):
            self._impl = obj._impl
            self._shape, self._dtype = obj._shape, obj._dtype

        elif isinstance(obj, ak.Array):
            self._impl = obj
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        elif hasattr(obj, "__dlpack_device__") and getattr(obj, "shape", None) == ():
            device_type, _ = obj.__dlpack_device__()
            if (
                isinstance(device_type, enum.Enum) and device_type.value == 1
            ) or device_type == 1:
                self._impl = np.array(obj)
                self._shape, self._dtype = (), self._impl.dtype
            elif (
                isinstance(device_type, enum.Enum) and device_type.value == 2
            ) or device_type == 2:
                cp = _import.cupy()
                self._impl = cp.array(obj)
                self._shape, self._dtype = (), self._impl.dtype
            else:
                msg = f"unsupported __dlpack_device__ type: {device_type}"
                raise TypeError(msg)

        elif isinstance(obj, (bool, numbers.Complex)):
            self._impl = np.array(obj)
            self._shape, self._dtype = (), self._impl.dtype

        else:
            self._impl = ak.Array(obj)
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)

        if dtype is not None and dtype != self._dtype:
            if isinstance(self._impl, ak.Array):
                self._impl = ak.values_astype(self._impl, dtype)
                self._shape, self._dtype = _shape_dtype(self._impl.layout)
            else:
                self._impl = np.array(obj, dtype=dtype)
                self._dtype = dtype

        if self._dtype.fields is not None:
            msg = f"dtype must not have fields: dtype.fields = {self._dtype.fields}"
            raise TypeError(msg)

        if self._dtype.shape != ():
            msg = f"dtype must not have a shape: dtype.shape = {self._dtype.shape}"
            raise TypeError(msg)

        if self._dtype.type not in numeric_types:
            msg = f"dtype must be numeric (bool, [u]int*, float*, complex*): dtype.type = {self._dtype.type}"
            raise TypeError(msg)

        if device is not None:
            if isinstance(self._impl, ak.Array) and device != ak.backend(self._impl):
                self._impl = ak.to_backend(self._impl, device)
            elif isinstance(self._impl, np.ndarray) and device == "cuda":
                cp = _import.cupy()
                self._impl = cp.array(self._impl)

        assert copy is None, "TODO"

    def __str__(self) -> str:
        """
        String representation of the array.
        """

        if len(self._shape) == 0:
            return f"{self._impl}"
        elif len(self._shape) == 1:
            return f"{ak._prettyprint.valuestr(self._impl, 1, 80)}"
        else:
            prep = ak._prettyprint.valuestr(self._impl, 20, 80 - 4)[1:-1].replace(
                "\n ", "\n    "
            )
            return f"[\n    {prep}\n]"

    def __repr__(self) -> str:
        """
        REPL-string representation of the array.
        """

        if len(self._shape) == 0:
            return f"ragged.array({self._impl})"
        elif len(self._shape) == 1:
            return f"ragged.array({ak._prettyprint.valuestr(self._impl, 1, 80 - 14)})"
        else:
            prep = ak._prettyprint.valuestr(self._impl, 20, 80 - 4)[1:-1].replace(
                "\n ", "\n    "
            )
            return f"ragged.array([\n    {prep}\n])"

    # Attributes: https://data-apis.org/array-api/latest/API_specification/array_object.html#attributes

    @property
    def dtype(self) -> Dtype:
        """
        Data type of the array elements.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.dtype.html
        """

        return self._dtype

    @property
    def device(self) -> Device:
        """
        Hardware device the array data resides on.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.device.html
        """

        return self._device

    @property
    def mT(self) -> array:
        """
        Transpose of a matrix (or a stack of matrices).

        Raises:
            ValueError: If any ragged dimension's lists are not sorted from longest
                to shortest, which is the only way that left-aligned ragged
                transposition is possible.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.mT.html
        """

        assert False, "TODO 1"

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.ndim.html
        """

        return len(self._shape)

    @property
    def shape(self) -> Shape:
        """
        Array dimensions.

        Regular dimensions are represented by `int` values in the `shape` and
        irregular (ragged) dimensions are represented by `None`.

        According to the specification, "An array dimension must be `None` if
        and only if a dimension is unknown," which is a different
        interpretation than we are making here.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.shape.html
        """

        return self._shape

    @property
    def size(self) -> None | int:
        """
        Number of elements in an array.

        This property never returns `None` because we do not consider
        dimensions to be unknown, and numerical values within ragged
        lists can be counted.

        Example:
            An array like `ragged.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])` has
            a size of 5 because it contains 5 numerical values.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html
        """

        if len(self._shape) == 0:
            return 1
        else:
            return int(ak.count(self._impl))

    @property
    def T(self) -> array:
        """
        Transpose of the array.

        Raises:
            ValueError: If any ragged dimension's lists are not sorted from longest
                to shortest, which is the only way that left-aligned ragged
                transposition is possible.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.T.html
        """

        assert False, "TODO 2"

    # methods: https://data-apis.org/array-api/latest/API_specification/array_object.html#methods

    def __abs__(self) -> array:
        """
        Calculates the absolute value for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__abs__.html
        """

        assert False, "TODO 3"

    def __add__(self, other: int | float | array, /) -> array:
        """
        Calculates the sum for each element of an array instance with the
        respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__add__.html
        """

        assert False, "TODO 4"

    def __and__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i & other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__and__.html
        """

        assert False, "TODO 5"

    def __array_namespace__(self, *, api_version: None | str = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html
        """

        assert api_version, "TODO"
        assert False, "TODO 6"

    def __bool__(self) -> bool:  # FIXME pylint: disable=E0304
        """
        Converts a zero-dimensional array to a Python `bool` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__bool__.html
        """

        return bool(self._impl)

    def __complex__(self) -> complex:
        """
        Converts a zero-dimensional array to a Python `complex` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__complex__.html
        """

        assert False, "TODO 8"

    def __dlpack__(self, *, stream: None | int | Any = None) -> PyCapsule:
        """
        Exports the array for consumption by `from_dlpack()` as a DLPack
        capsule.

        Args:
            stream: CuPy Stream object (https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
                if not `None`.

        Raises:
            ValueError: If any dimensions are ragged.

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
        """

        assert stream, "TODO"
        assert False, "TODO 9"

    def __dlpack_device__(self) -> tuple[enum.Enum, int]:
        """
        Returns device type and device ID in DLPack format.

        Raises:
            ValueError: If any dimensions are ragged.

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html
        """

        assert False, "TODO 10"

    def __eq__(self, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        """
        Computes the truth value of `self_i == other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__eq__.html
        """

        assert False, "TODO 11"

    def __float__(self) -> float:
        """
        Converts a zero-dimensional array to a Python `float` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__float__.html
        """

        assert False, "TODO 12"

    def __floordiv__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i // other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__floordiv__.html
        """

        assert False, "TODO 13"

    def __ge__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i >= other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ge__.html
        """

        assert False, "TODO 14"

    def __getitem__(self, key: GetSliceKey, /) -> array:
        """
        Returns self[key].

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__getitem__.html
        """

        assert False, "TODO 15"

    def __gt__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i > other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__gt__.html
        """

        assert False, "TODO 16"

    def __index__(self) -> int:  # FIXME pylint: disable=E0305
        """
        Converts a zero-dimensional integer array to a Python `int` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__index__.html
        """

        assert False, "TODO 17"

    def __int__(self) -> int:
        """
        Converts a zero-dimensional array to a Python `int` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__int__.html
        """

        assert False, "TODO 18"

    def __invert__(self) -> array:
        """
        Evaluates `~self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__invert__.html
        """

        assert False, "TODO 19"

    def __le__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i <= other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__le__.html
        """

        assert False, "TODO 20"

    def __lshift__(self, other: int | array, /) -> array:
        """
        Evaluates `self_i << other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__lshift__.html
        """

        assert False, "TODO 21"

    def __lt__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i < other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__lt__.html
        """

        assert False, "TODO 22"

    def __matmul__(self, other: array, /) -> array:
        """
        Computes the matrix product.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__matmul__.html
        """

        assert False, "TODO 22"

    def __mod__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i % other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__mod__.html
        """

        assert False, "TODO 23"

    def __mul__(self, other: int | float | array, /) -> array:
        """
        Calculates the product for each element of an array instance with the
        respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__mul__.html
        """

        assert False, "TODO 24"

    def __ne__(self, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        """
        Computes the truth value of `self_i != other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ne__.html
        """

        assert False, "TODO 25"

    def __neg__(self) -> array:
        """
        Evaluates `-self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__neg__.html
        """

        assert False, "TODO 26"

    def __or__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i | other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__or__.html
        """

        assert False, "TODO 27"

    def __pos__(self) -> array:
        """
        Evaluates `+self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__pos__.html
        """

        assert False, "TODO 28"

    def __pow__(self, other: int | float | array, /) -> array:
        """
        Calculates an implementation-dependent approximation of exponentiation
        by raising each element (the base) of an array instance to the power of
        `other_i` (the exponent), where `other_i` is the corresponding element
        of the array `other`.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__pow__.html
        """

        assert False, "TODO 29"

    def __rshift__(self, other: int | array, /) -> array:
        """
        Evaluates `self_i >> other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__rshift__.html
        """

        assert False, "TODO 30"

    def __setitem__(
        self, key: SetSliceKey, value: int | float | bool | array, /
    ) -> None:
        """
        Sets `self[key]` to value.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__setitem__.html
        """

        assert False, "TODO 31"

    def __sub__(self, other: int | float | array, /) -> array:
        """
        Calculates the difference for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__sub__.html
        """

        assert False, "TODO 32"

    def __truediv__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i / other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__truediv__.html
        """

        assert False, "TODO 33"

    def __xor__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i ^ other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__xor__.html
        """

        assert False, "TODO 34"

    def to_device(self, device: Device, /, *, stream: None | int | Any = None) -> array:
        """
        Copy the array from the device on which it currently resides to the
        specified device.

        Args:
            device: If `"cpu"`, the array is backed by NumPy and resides in
                main memory; if `"cuda"`, the array is backed by CuPy and
                resides in CUDA global memory.
            stream: CuPy Stream object (https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
                for `device="cuda"`.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.to_device.html
        """

        if isinstance(self._impl, ak.Array):
            if device != ak.backend(self._impl):
                assert stream is None, "TODO"
                impl = ak.to_backend(self._impl, device)
            else:
                impl = self._impl

        elif isinstance(self._impl, np.ndarray):
            # self._impl is a NumPy 0-dimensional array
            if device == "cuda":
                assert stream is None, "TODO"
                cp = _import.cupy()
                impl = cp.array(self._impl)
            else:
                impl = self._impl

        else:
            # self._impl is a CuPy 0-dimensional array
            impl = self._impl.get() if device == "cpu" else self._impl  # type: ignore[union-attr]

        return self._new(impl, self._shape, self._dtype, device)

    # in-place operators: https://data-apis.org/array-api/2022.12/API_specification/array_object.html#in-place-operators

    def __iadd__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self + other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self + other
        self._impl, self._device = out._impl, out._device
        return self

    def __isub__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self - other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self - other
        self._impl, self._device = out._impl, out._device
        return self

    def __imul__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self * other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self * other
        self._impl, self._device = out._impl, out._device
        return self

    def __itruediv__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self / other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self / other
        self._impl, self._device = out._impl, out._device
        return self

    def __ifloordiv__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self // other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self // other
        self._impl, self._device = out._impl, out._device
        return self

    def __ipow__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self ** other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self**other
        self._impl, self._device = out._impl, out._device
        return self

    def __imod__(self, other: int | float | array, /) -> array:
        """
        Calculates `self = self % other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self % other
        self._impl, self._device = out._impl, out._device
        return self

    def __imatmul__(self, other: array, /) -> array:
        """
        Calculates `self = self @ other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self @ other
        self._impl, self._device = out._impl, out._device
        return self

    def __iand__(self, other: int | bool | array, /) -> array:
        """
        Calculates `self = self & other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self & other
        self._impl, self._device = out._impl, out._device
        return self

    def __ior__(self, other: int | bool | array, /) -> array:
        """
        Calculates `self = self | other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self | other
        self._impl, self._device = out._impl, out._device
        return self

    def __ixor__(self, other: int | bool | array, /) -> array:
        """
        Calculates `self = self ^ other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self ^ other
        self._impl, self._device = out._impl, out._device
        return self

    def __ilshift__(self, other: int | array, /) -> array:
        """
        Calculates `self = self << other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self << other
        self._impl, self._device = out._impl, out._device
        return self

    def __irshift__(self, other: int | array, /) -> array:
        """
        Calculates `self = self >> other` in-place.

        (Internal arrays are immutable; this only replaces the array that the
        Python object points to.)
        """

        out = self >> other
        self._impl, self._device = out._impl, out._device
        return self

    # reflected operators: https://data-apis.org/array-api/2022.12/API_specification/array_object.html#reflected-operators

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__
    __rpow__ = __pow__
    __rmod__ = __mod__
    __rmatmul__ = __matmul__
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__
    __rlshift__ = __lshift__
    __rrshift__ = __rshift__


def _unbox(*inputs: array) -> tuple[ak.Array | SupportsDLPack, ...]:
    if len(inputs) > 1 and any(type(inputs[0]) is not type(x) for x in inputs):
        types = "\n".join(f"{type(x).__module__}.{type(x).__name__}" for x in inputs)
        msg = f"mixed array types: {types}"
        raise TypeError(msg)

    return tuple(x._impl for x in inputs)  # pylint: disable=W0212


def _box(
    cls: type[array],
    output: ak.Array | np.number | SupportsDLPack,
    *,
    dtype: None | Dtype = None,
) -> array:
    if isinstance(output, ak.Array):
        impl = output
        shape, dtype_observed = _shape_dtype(output.layout)
        if dtype is not None and dtype != dtype_observed:
            impl = ak.values_astype(impl, dtype)
        else:
            dtype = dtype_observed
        device = ak.backend(output)

    elif isinstance(output, np.number):
        impl = np.array(output)
        shape = output.shape
        dtype_observed = output.dtype
        if dtype is not None and dtype != dtype_observed:
            impl = impl.astype(dtype)
        else:
            dtype = dtype_observed
        device = "cpu"

    else:
        impl = output
        shape = output.shape  # type: ignore[union-attr]
        dtype_observed = output.dtype  # type: ignore[union-attr]
        if dtype is not None and dtype != dtype_observed:
            impl = impl.astype(dtype)
        else:
            dtype = dtype_observed
        device = "cpu" if isinstance(output, np.ndarray) else "cuda"

    return cls._new(impl, shape, dtype, device)  # pylint: disable=W0212
