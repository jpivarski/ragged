# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Defines a ragged array module that is compliant with version 2022.12 of the
Array API.

This is the current default: `ragged.v202212.*` is imported into `ragged.*`.

https://data-apis.org/array-api/2022.12/API_specification/
"""

from __future__ import annotations

from ._obj import array

__all__ = ["array"]