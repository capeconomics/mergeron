"""Variables, types, objects and functions used throughout the package."""

from __future__ import annotations

import enum
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import attrs
import mpmath
import numpy as np
from ruamel import yaml

YAML = yaml.YAML(typ="rt")
YAML.indent(mapping=2, sequence=4, offset=2)


# Add functions for serializing/deserializing some objects used
# or defined in this package

# Add yaml representer, constructor for various types
# NoneType
(_, _) = (
    YAML.representer.add_representer(
        type(None), lambda _r, _d: _r.represent_scalar("!None", "none")
    ),
    YAML.constructor.add_constructor("!None", lambda _c, _n, /: None),
)

# Decimal
(_, _) = (
    YAML.representer.add_representer(
        Decimal, lambda _r, _d: _r.represent_scalar("!Decimal", f"{_d}")
    ),
    YAML.constructor.add_constructor(
        "!Decimal", lambda _c, _n, /: Decimal(_c.construct_scalar(_n))
    ),
)


# MappingProxyType
_, _ = (
    YAML.representer.add_representer(
        MappingProxyType,
        lambda _r, _d: _r.represent_mapping("!mappingproxy", dict(_d.items())),
    ),
    YAML.constructor.add_constructor(
        "!mappingproxy", lambda _c, _n: MappingProxyType(dict(**yaml_rt_mapper(_c, _n)))
    ),
)

# mpmpath.mpf
(_, _) = (
    YAML.representer.add_representer(
        mpmath.mpf, lambda _r, _d: _r.represent_scalar("!MPFloat", f"{_d}")
    ),
    YAML.constructor.add_constructor(
        "!MPFloat", lambda _c, _n, /: mpmath.mpf(_c.construct_scalar(_n))
    ),
)

# mpmath.matrix
(_, _) = (
    YAML.representer.add_representer(
        mpmath.matrix, lambda _r, _d: _r.represent_sequence("!MPMatrix", _d.tolist())
    ),
    YAML.constructor.add_constructor(
        "!MPMatrix",
        lambda _c, _n, /: mpmath.matrix(_c.construct_sequence(_n, deep=True)),
    ),
)

# set
(_, _) = (
    YAML.representer.add_representer(
        set, lambda _r, _d: _r.represent_sequence("!set", tuple(_d))
    ),
    YAML.constructor.add_constructor(
        "!set",
        lambda _c, _n, /: {tuple(_f) for _f in _c.construct_sequence(_n, deep=True)},
    ),
)


# nu.uint8
(_, _) = (
    YAML.representer.add_representer(
        np.uint8, lambda _r, _d: _r.represent_scalar("!uint8", f"{_d}")
    ),
    YAML.constructor.add_constructor(
        "!uint8", lambda _c, _n, /: np.ubyte(_c.construct_scalar(_n))
    ),
)

# np.ndarray
(_, _) = (
    YAML.representer.add_representer(
        np.ndarray,
        lambda _r, _d: _r.represent_sequence("!ndarray", (_d.tolist(), _d.dtype.str)),
    ),
    YAML.constructor.add_constructor(
        "!ndarray", lambda _c, _n, /: np.array(*_c.construct_sequence(_n, deep=True))
    ),
)


def yaml_rt_mapper(
    _c: yaml.constructor.RoundTripConstructor, _n: yaml.MappingNode
) -> Mapping[str, Any]:
    """Construct mapping from a mapping node with the RoundTripConstructor."""
    data_: Mapping[str, Any] = yaml.constructor.CommentedMap()
    _c.construct_mapping(_n, maptyp=data_, deep=True)
    return data_


PKG_ATTRS_MAP: dict[str, type] = {}


def yamlize_attrs(_typ: type, /, *, attr_map: dict[str, type] = PKG_ATTRS_MAP) -> None:
    """Add yaml representer, constructor for attrs-defined class.

    Attributes with property, `init=False` are not serialized/deserialized
    to YAML by the functions defined here. These attributes can, of course,
    be dumped to stand-alone (YAML) representation, and deserialized from there.
    """
    if not attrs.has(_typ):
        raise ValueError(f"Object {_typ} is not attrs-defined")

    _typ_tag = f"!{_typ.__name__}"
    attr_map |= {_typ_tag: _typ}

    _ = YAML.representer.add_representer(
        _typ,
        lambda _r, _d: _r.represent_mapping(
            _typ_tag,
            {_a.name: getattr(_d, _a.name) for _a in _d.__attrs_attrs__ if _a.init},
        ),
    )
    _ = YAML.constructor.add_constructor(
        _typ_tag, lambda _c, _n: attr_map[_typ_tag](**yaml_rt_mapper(_c, _n))
    )


# Yamelized Enum
@YAML.register_class
class Enameled(enum.Enum):
    """Add YAML representer, constructor for enum.Enum."""

    def __str__(self) -> Any:
        """Customize the string representation for f-string usage."""
        return f"{self.value}"

    @classmethod
    def to_yaml(
        cls, _r: yaml.representer.RoundTripRepresenter, _d: enum.Enum
    ) -> yaml.ScalarNode:
        """Serialize enumerations by .name, not .value."""
        return _r.represent_scalar(
            f"!{super().__getattribute__(cls, '__name__')}", f"{_d.name}"
        )

    @classmethod
    def from_yaml(
        cls, _c: yaml.constructor.RoundTripConstructor, _n: yaml.ScalarNode
    ) -> enum.EnumType:
        """Deserialize enumeration serialized by .name."""
        retval: enum.EnumMeta = super().__getattribute__(cls, _n.value)
        return retval
