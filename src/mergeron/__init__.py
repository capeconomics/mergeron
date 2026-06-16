"""Variables, types, objects and functions used throughout the package."""

from __future__ import annotations

import enum
import os
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias

import numpy as np
from mpmath import ctx_mp_python
from mpmath.matrices import matrices

from ._serialization import YAML as YAML
from ._serialization import Enameled as Enameled
from ._serialization import yaml_rt_mapper as yaml_rt_mapper
from ._serialization import yamlize_attrs as yamlize_attrs

if sys.version_info >= (3, 14):
    import zipfile as zipfile_conditional
else:
    from backports.zstd import zipfile as zipfile_conditional

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ruamel import yaml

# Need the two-step re-export for sphinx/autoapi
zipfile = zipfile_conditional

VERSION = "2026.739782.1"

__version__ = VERSION

PKG_NAME: str = Path(__file__).parent.name

WORK_DIR: Path = Path(os.getenv("MERGERON_WORK_DIR", f"{Path.home() / PKG_NAME}"))
"""
If defined, the global variable WORK_DIR is used as a data store.

If the user does not define WORK_DIR, a subdirectory in
the user's home directory, having the name of this package, is
created/reused.
"""
if not WORK_DIR.is_dir():
    WORK_DIR.mkdir(parents=False)

DEFAULT_REC = float(os.getenv("MERGERON_DEFAULT_REC", "0.85"))
"""
Default recapture rate.

Can be overridden by setting the environment variable
MERGERON_DEFAULT_REC to a value between 0 and 1.
"""

NTHREADS = int(os.getenv("MERGERON_NTHREADS", "32"))

MPFloat: TypeAlias = ctx_mp_python._mpf
MPMatrix: TypeAlias = matrices._matrix

np.set_printoptions(precision=28, floatmode="fixed", legacy=False)


@YAML.register_class
@enum.unique
class RECForm(str, Enameled):
    R"""For derivation of recapture rate from market shares.

    With :math:`\mathscr{N}` a set of firms, each supplying a
    single differentiated product, and :math:`\mathscr{M} \subset \mathscr{N}`
    a putative relevant product market, with
    :math:`d_{ij}` denoting diversion ratio from good :math:`i` to good :math:`j`,
    :math:`s_i` denoting market shares, and
    :math:`\overline{r}` the default market recapture rate,
    market recapture rates for the respective products may be specified
    as having one of the following forms:
    """

    FIXED = "fixed"
    R"""Given, :math:`\overline{r}`,

    .. math::

        REC_i = \overline{r} {\ } \forall {\ } i \in \mathscr{M}

    """

    INOUT = "inside-out"
    R"""
    Given, :math:`\overline{r}, s_i {\ } \forall {\ } i \in \mathscr{M}`, with
    :math:`s_{min} = \min(s_1, s_2)`,

    .. math::

        REC_i = \frac{\overline{r} (1 - s_i)}{1 - (1 - \overline{r}) s_{min} - \overline{r} s_i}
        {\ } \forall {\ } i \in \mathscr{M}

    The default within this package.
    """

    OUTIN = "outside-in"
    R"""
    Given, :math:`d_{ij} {\ } \forall {\ } i, j \in \mathscr{M}, i \neq j`,

    .. math::

        REC_i = {\sum_{j \in \mathscr{M}}^{j \neq i} d_{ij}}
        {\ } \forall {\ } i \in \mathscr{M}

    """


@YAML.register_class
@enum.unique
class UPPAggregator(str, Enameled):
    """Aggregator for GUPPI and diversion ratio estimates."""

    AVG = "average"
    CPA = "cross-product-share weighted average"
    CPD = "cross-product-share weighted distance"
    CPG = "cross-product-share weighted geometric mean"
    DIS = "symmetrically-weighted distance"
    GMN = "geometric mean"
    MAX = "max"
    MIN = "min"
    OSA = "own-share weighted average"
    OSD = "own-share weighted distance"
    OSG = "own-share weighted geometric mean"


# https://numpy.org/devdocs/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
@YAML.register_class
class TypedNDArray(np.ndarray):
    """NDArray with type specification."""

    def __new__(  # noqa: D102
        cls,
        _arr: Sequence[
            bool | float | int | MPFloat | Sequence[bool | float | int | MPFloat]
        ]
        | NDArray[np.bool | np.floating | np.integer | np.uint8]
        | MPMatrix,
        _type: bool
        | float
        | int
        | MPFloat
        | np.bool
        | np.floating
        | np.integer
        | np.uint8,
        /,
        *,
        info: dict[str, Any] | None = None,
    ) -> TypedNDArray:
        if not hasattr(_arr, "__len__") or isinstance(_arr, str):
            raise ValueError(f"Invalid first argument, {_arr!r}")

        if _type in {np.floating, np.integer}:
            _dtype = type(np.ravel(_arr)[0])
        elif isinstance(_arr, MPMatrix):  # type: ignore[misc]
            _dtype = object
            _arr = _arr.tolist()
        elif _type is MPFloat:
            _dtype = object
        else:
            _dtype = _type

        if not len(np.ravel(_arr)):
            _arr = np.array([], dtype=_dtype)  # type: ignore[arg-type]
        elif (
            np.asarray(_arr).shape
            and isinstance(_arr[0], Sequence | np.ndarray)
            and isinstance(np.ravel(_arr)[0], MPFloat)  # type: ignore[misc]
            and not ((_dtype is object) or np.issubdtype(_dtype, np.floating))  # type: ignore[arg-type]
        ):
            raise ValueError(
                "Array of MPFloat objects can only be converted to "
                f"ArrayFloat, ArrayDouble, or ArrayMPFloat, not {_type!r}."
            )
        elif not np.issubdtype((_dt := np.asarray(_arr, _dtype).dtype), _dtype):  # type: ignore[arg-type]
            raise ValueError(
                f"Array type, {_dt!r}, is not compatible with target dtype, {_type}."
            )
        obj = np.asarray(_arr, _dtype).view(cls)  # type: ignore[arg-type]
        obj.info = info
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:  # noqa: D105
        if obj is None:
            return
        self.info = getattr(obj, "info", None)
        return

    @classmethod
    def to_yaml(
        cls, _r: yaml.representer.RoundTripRepresenter, _d: np.ndarray
    ) -> yaml.SequenceNode:
        """Serialize TypedNDArray."""
        return _r.represent_sequence(
            f"!{super().__getattribute__(cls, '__name__')}", (_d.tolist(), _d.dtype.str)
        )

    @classmethod
    def from_yaml(
        cls, _c: yaml.constructor.RoundTripConstructor, _n: yaml.SequenceNode
    ) -> TypedNDArray:
        """Deserialize TypedNDArray."""
        _data = np.asarray(*_c.construct_sequence(_n, deep=True))
        return TypedNDArray.__new__(cls, _data, _data.dtype).view(cls)


@YAML.register_class
class ArrayBoolean(TypedNDArray):
    """Array of booleans."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[bool | Sequence[bool]] | NDArray[np.bool_]
    ) -> ArrayBoolean:
        return super().__new__(cls, _arr, np.bool).view(cls)


@YAML.register_class
class ArrayDouble(TypedNDArray):
    """Array of double-precision floats."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[float | Sequence[float]] | NDArray[np.float64]
    ) -> ArrayDouble:
        return super().__new__(cls, _arr, np.float64).view(cls)


@YAML.register_class
class ArrayFloat(TypedNDArray):
    """Array of floats, any precision supported by numpy."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[float | Sequence[float]] | NDArray[np.floating]
    ) -> ArrayFloat:
        _dtype = _arr.dtype if hasattr(_arr, "dtype") else np.float64
        if not np.issubdtype(_dtype, np.floating):
            raise ValueError(f"Array type, {_dtype!r}, is not a float type.")
        return super().__new__(cls, _arr, _dtype).view(cls)


@YAML.register_class
class ArrayBIGINT(TypedNDArray):
    """Array of 64-bit integers."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[int | Sequence[int]] | NDArray[np.int64]
    ) -> ArrayBIGINT:
        return super().__new__(cls, _arr, np.int64).view(cls)


@YAML.register_class
class ArrayINT(TypedNDArray):
    """Array of integers, any precision supported by numpy."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[int | Sequence[int]] | NDArray[np.integer]
    ) -> ArrayINT:
        _dtype = _arr.dtype if hasattr(_arr, "dtype") else np.int64
        if not np.issubdtype(_dtype, np.integer):
            raise ValueError(f"Array type, {_dtype!r}, is not an integer type.")
        return super().__new__(cls, _arr, _dtype).view(cls)


@YAML.register_class
class ArrayMPFloat(TypedNDArray):
    """Array of arbitrary-precision floats, mpmath.mpf()s."""

    def __new__(  # noqa: D102
        cls, _arr: Sequence[MPFloat | Sequence[MPFloat]] | np.ndarray
    ) -> ArrayMPFloat:
        if not hasattr(next(np.asarray(_arr).flat), "mpf_convert_arg"):
            raise ValueError("Data array cannot be converted to ArrayMPFloat.")
        return super().__new__(cls, _arr, np.object_).view(cls)


@YAML.register_class
class ArrayUINT8(TypedNDArray):
    """Array of 8-bit unsigned integers."""

    def __new__(cls, _arr: NDArray[np.uint8]) -> ArrayUINT8:  # noqa: D102
        return super().__new__(cls, _arr, np.uint8).view(cls)


EMPTY_ARRAYBIGINT = ArrayBIGINT(np.array([], int))
EMPTY_ARRAYBOOLEAN = ArrayBoolean(np.array([], bool))
EMPTY_ARRAYDOUBLE = ArrayDouble(np.array([], float))
EMPTY_ARRAYUINT8 = ArrayUINT8(np.array([], np.uint8))


# some functions useful for transforming python data structures
def invert_map(_dict: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Invert mapping, mapping values to keys of the original mapping."""
    return {_v: _k for _k, _v in _dict.items()}


def _dict_from_mapping(_p: Mapping[Any, Any], /) -> dict[Any, Any]:
    retval: dict[Any, Any] = {}
    for _k, _v in _p.items():
        retval |= {_k: _dict_from_mapping(_v)} if isinstance(_v, Mapping) else {_k: _v}
    return retval


def _mappingproxy_from_mapping(_p: Mapping[Any, Any], /) -> MappingProxyType[Any, Any]:
    retval: dict[Any, Any] = {}
    for _k, _v in _p.items():
        retval |= (
            {_k: _mappingproxy_from_mapping(_v)}
            if isinstance(_v, Mapping)
            else {_k: _v}
        )
    return MappingProxyType(retval)
