# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Generic definitions used by the version-specific modules, such as
`ragged.v202212`.

https://data-apis.org/array-api/latest/API_specification/
"""

from __future__ import annotations

from ._const import (
    e,
    inf,
    nan,
    newaxis,
    pi,
)
from ._creation import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from ._datatype import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)
from ._elementwise import (  # pylint: disable=W0622
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    conj,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    imag,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
)
from ._indexing import (
    take,
)
from ._linalg import (
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)
from ._manipulation import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)
from ._obj import array
from ._search import (
    argmax,
    argmin,
    nonzero,
    where,
)
from ._set import (
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)
from ._sorting import (
    argsort,
    sort,
)
from ._statistical import (  # pylint: disable=W0622
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)
from ._utility import (  # pylint: disable=W0622
    all,
    any,
)

__all__ = [
    # _const
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
    # _creation
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    # _datatype
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
    # _elementwise
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    # _indexing
    "take",
    # _linalg
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
    # _manipulation
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    # _obj
    "array",
    # _search
    "argmax",
    "argmin",
    "nonzero",
    "where",
    # _set
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    # _sorting
    "argsort",
    "sort",
    # _statistical
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
    # _utility
    "all",
    "any",
]
