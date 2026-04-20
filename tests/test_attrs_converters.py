"""Test yaml representers and converters for attrs-generated classes."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from attrs import Attribute
from attrs import Converter
from attrs import cmp_using
from attrs import field
from attrs import frozen
from attrs import validators
from ruamel import yaml
from ruamel.yaml.compat import StringIO

import mergeron.gen.data_generation as dgm
from mergeron import DEFAULT_REC
from mergeron import YAML
from mergeron import ArrayFloat
from mergeron import ArrayINT
from mergeron import RECForm
from mergeron.core import DEFAULT_DIST_PARMS
from mergeron.gen import DEFAULT_FCOUNT_WTS
from mergeron.gen import SHRDistribution

if TYPE_CHECKING:
    from pathlib import Path


def _dump_to_yaml_str(
    _yaml: yaml.YAML,
    _object: Any,
    _stream: Path | StringIO | None = None,
    **kw: Mapping[str, Any],
) -> Any | None:
    if _stream is None:
        _stream = StringIO()

    _yaml.dump(_object, _stream, **kw)
    if isinstance(_stream, StringIO):
        return _stream.getvalue()
    else:
        return None


def _fc_wts_conv(
    _v: Sequence[float | int] | ArrayFloat | ArrayINT | None, _i: ExampleMarketShareSpec
) -> ArrayFloat | None:
    if _i.distribution == SHRDistribution.UNI:
        return None
    elif _v is None or len(_v) == 0 or np.array_equal(_v, DEFAULT_FCOUNT_WTS):
        return DEFAULT_FCOUNT_WTS
    else:
        return _tv if (_tv := np.asarray(_v, float)).sum() == 1 else _tv / _tv.sum()


def _shr_dp_conv(
    _v: Sequence[float] | ArrayFloat | None, _i: ExampleMarketShareSpec
) -> ArrayFloat:
    match _v:
        case None if _i.distribution == SHRDistribution.UNI:
            retval = DEFAULT_DIST_PARMS
        case None:
            fc_max = 1 + (
                len(DEFAULT_FCOUNT_WTS)
                if not hasattr(_i, "firm_counts_weights")
                or _i.firm_counts_weights is None
                or len(_i.firm_counts_weights) == 0
                else len(_i.firm_counts_weights)
            )

            match _i.distribution:
                case SHRDistribution.DIR_FLAT | SHRDistribution.DIR_FLAT_CONSTR:
                    retval = np.ones(fc_max, float)
                case SHRDistribution.DIR_ASYM:
                    retval = np.array([2.0] * 6 + [1.5] * 5 + [1.25] * fc_max, float)
                case SHRDistribution.DIR_COND:
                    retval = np.array([], float)
                case _ if isinstance(_i.distribution, SHRDistribution):
                    raise ValueError(
                        f"No default defined for market share distribution, {_i.distribution!r}"
                    )
                case _:
                    raise ValueError(
                        f"Unsupported distribution for market share generation, {_i.distribution!r}"
                    )
        case _ if isinstance(_v, Sequence | np.ndarray):
            retval = np.asarray(_v, float)
        case _:
            raise ValueError(
                f"Input, {_v!r} has invalid type. Must be None, Sequence of floats, or Numpy ndarray."
            )

    return ArrayFloat(retval)


@YAML.register_class
@frozen
class ExampleMarketShareSpec:
    """Market share specification.

    A salient feature of market-share specification in this package is that
    the draws represent markets with multiple different firm-counts.
    Firm-counts are unspecified if the share distribution is
    :attr:`mergeron.SHRDistribution.UNI`, for Dirichlet-distributed market-shares,
    the default specification is that firm-counts  vary between
    2 and 7 firms with each value equally likely.

    Notes
    -----
    If :attr:`mergeron.gen.ShareSpec.distribution` == :attr:`mergeron.gen.SHRDistribution.UNI`,
    then it is infeasible that
    :attr:`mergeron.gen.ShareSpec.recapture_form` == :attr:`mergeron.RECForm.OUTIN`.
    In other words, if the distribution of markets over firm-counts is unspecified,
    recapture rates cannot be estimated using outside-good choice probabilities.

    For a sample with explicit firm counts, market shares must be specified as
    having a supported Dirichlet distribution (see :class:`mergeron.gen.SHRDistribution`).

    """

    distribution: SHRDistribution = field(
        kw_only=False, validator=validators.instance_of(SHRDistribution)
    )
    """See :class:`SHRDistribution`"""

    firm_counts_weights: ArrayFloat | None = field(
        kw_only=True,
        eq=cmp_using(eq=np.array_equal),
        converter=Converter(_fc_wts_conv, takes_self=True),  # type: ignore
    )
    """Relative or absolute frequencies of firm counts

    Given frequencies are exogenous to generated market data sample;
    defaults to DEFAULT_FCOUNT_WTS, which specifies
    firm-counts of 2 to 6 with weights in descending order from 5 to 1.

    This parameter does not apply for market shares with Uniform distribution; it
    is forced to None if
    :attr:`mergeron.gen.ShareSpec.distribution` == :attr:`mergeron.gen.SHRDistribution.UNI`.
    """

    @firm_counts_weights.default
    def _fcwd(_i: ExampleMarketShareSpec) -> ArrayFloat | None:
        return _fc_wts_conv(None, _i)

    @firm_counts_weights.validator
    def _fcv(
        _i: ExampleMarketShareSpec, _a: Attribute[ArrayFloat], _v: ArrayFloat
    ) -> None:
        if _i.distribution != SHRDistribution.UNI and not len(_v):  # any(_v.shape)
            raise ValueError(
                f"Attribute, {'"firm_counts_weights"'} must not be empty except "
                "when shares have uniform distribution."
            )

    parameters: ArrayFloat = field(
        kw_only=True,
        eq=cmp_using(eq=np.array_equal),
        converter=Converter(_shr_dp_conv, takes_self=True),  # type: ignore
    )
    """Parameters for tailoring market-share distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Dirichlet-type distributions, a vector of shape parameters of length
    no less than the length of firm-count weights below; defaults depend on
    type of Dirichlet-distribution specified.

    """

    @parameters.default
    def _dpd(_i: ExampleMarketShareSpec) -> ArrayFloat:
        # converters run after defaults, and we
        # avoid redundancy and confusion here
        return _shr_dp_conv(None, _i)

    @parameters.validator
    def _dpv(
        _i: ExampleMarketShareSpec, _a: Attribute[ArrayFloat], _v: ArrayFloat
    ) -> None:
        if (
            _i.firm_counts_weights is not None
            and _v is not None
            and 0 < len(_v) < (1 + len(_i.firm_counts_weights))
        ):
            raise ValueError(
                "If specified, the number of distribution parameters must euqal or "
                "exceed the maximum firm-count premerger, which is "
                "1 plus the length of the vector specifying firm-count weights."
            )

    recapture_form: RECForm = field(
        kw_only=True, default=RECForm.INOUT, validator=validators.instance_of(RECForm)
    )
    """See :class:`mergeron.RECForm`"""

    @recapture_form.validator
    def _rfv(_i: ExampleMarketShareSpec, _a: Attribute[RECForm], _v: RECForm) -> None:
        if _i.distribution == SHRDistribution.UNI and _v == RECForm.OUTIN:
            raise ValueError(
                "Outside-good choice probabilities cannot be generated if the distribution of "
                "markets over firm-counts is unspecified."
            )

    recapture_rate: float | None = field(kw_only=True)
    """A value between 0 and 1.

    :code:`None` if market share specification requires direct generation of
    outside good choice probabilities (:attr:`mergeron.RECForm.OUTIN`).

    The recapture rate is usually calibrated to the numbers-equivalent of the
    HHI threshold for the presumtion of harm from unilateral competitive effects
    in published merger guidelines. Accordingly, the recapture rate rounded to
    the nearest 5% is:

    * 0.85, **7-to-6 merger from symmetry**; US Guidelines, 1992, 2023
    * 0.80, 5-to-4 merger from symmetry
    * 0.80, **5-to-4 merger to symmetry**; US Guidelines, 2010

    Highlighting indicates hypothetical mergers in the neighborhood of (the boundary of)
    the Guidelines presumption of harm. (In the EU Guidelines, concentration measures serve as
    screens for further investigation, rather than as the basis for presumptions of harm or
    presumptions no harm.)

    ALERT: If diversion ratios are estimated by specifying a choice probability for the
    outside good, the recapture rate is set to None, overriding any user-specified value.

    """

    @recapture_rate.default
    def _rrd(_i: ExampleMarketShareSpec) -> float | None:
        return None if _i.recapture_form == RECForm.OUTIN else DEFAULT_REC

    @recapture_rate.validator
    def _rrv(_i: ExampleMarketShareSpec, _a: Attribute[float], _v: float) -> None:
        if _v and not (0 < _v <= 1):
            raise ValueError("Recapture rate must lie in the interval, [0, 1).")
        elif _v is None and _i.recapture_form != RECForm.OUTIN:
            raise ValueError(
                f"Recapture specification, {_i.recapture_form!r} requires that "
                "the market sample specification inclues a recapture rate in the "
                "interval [0, 1)."
            )

    @classmethod
    def to_yaml(
        cls, _r: yaml.representer.RoundTripRepresenter, _d: ExampleMarketShareSpec
    ) -> yaml.MappingNode:
        retval: yaml.MappingNode = _r.represent_mapping(
            f"!{cls.__name__}",
            {_a.name: getattr(_d, _a.name) for _a in _d.__attrs_attrs__},
        )
        return retval

    @classmethod
    def from_yaml(
        cls, _c: yaml.constructor.RoundTripConstructor, _n: yaml.MappingNode
    ) -> ExampleMarketShareSpec:
        data_ = yaml.constructor.CommentedMap()
        _c.construct_mapping(_n, maptyp=data_)
        return cls(**data_)


def test_share_spec_default() -> None:
    """Test the default market share specification."""
    share_spec = ExampleMarketShareSpec(dgm.SHRDistribution.UNI)
    if share_spec.distribution != dgm.SHRDistribution.UNI:
        raise ValueError(
            f"Expected {dgm.SHRDistribution.UNI!r}, got {share_spec.distribution!r}."
        )
    elif np.array_repr(share_spec.parameters) != np.array_repr(DEFAULT_DIST_PARMS):
        raise ValueError(
            f"Expected {DEFAULT_DIST_PARMS!r}, got {share_spec.parameters!r}."
        )
    elif share_spec.firm_counts_weights is not None:
        raise ValueError(f"Expected {None!r}, got {share_spec.firm_counts_weights!r}.")
    elif share_spec.recapture_form != RECForm.INOUT:
        raise ValueError(
            f"Expected {RECForm.INOUT!r}, got {share_spec.recapture_form!r}."
        )
    elif share_spec.recapture_rate != DEFAULT_REC:
        raise ValueError(
            f"Expected {DEFAULT_REC!r}, got {share_spec.recapture_rate!r}."
        )


def test_share_spec_dir_default() -> None:
    """Test the default market share specification with Dirichlet distribution."""
    share_spec = ExampleMarketShareSpec(
        dgm.SHRDistribution.DIR_FLAT, firm_counts_weights=DEFAULT_FCOUNT_WTS
    )
    if share_spec.distribution != dgm.SHRDistribution.DIR_FLAT:
        raise ValueError(
            f"Expected {dgm.SHRDistribution.DIR_FLAT!r}, got {share_spec.distribution!r}."
        )
    elif not np.array_equal(
        share_spec.parameters, np.ones(1 + len(DEFAULT_FCOUNT_WTS))
    ):
        print(repr(share_spec))
        raise ValueError(f"Got {share_spec.parameters!r}.")
    elif not np.array_equal(
        np.asarray(share_spec.firm_counts_weights), DEFAULT_FCOUNT_WTS
    ):
        print(repr(share_spec))
        print(
            repr(
                np.array_equal(
                    DEFAULT_FCOUNT_WTS, np.asarray(share_spec.firm_counts_weights)
                )
            )
        )
        raise ValueError(f"got {share_spec.firm_counts_weights!r}.")
    elif share_spec.recapture_form != RECForm.INOUT:
        raise ValueError(f"got {share_spec.recapture_form!r}.")
    elif share_spec.recapture_rate != DEFAULT_REC:
        raise ValueError(f"got {share_spec.recapture_rate!r}.")


def test_local_yaml_test() -> None:
    """Test serializing-deserializing a market share specification."""
    test_share_spec = ExampleMarketShareSpec(dgm.SHRDistribution.UNI)

    yaml_str = _dump_to_yaml_str(YAML, test_share_spec)
    try:
        test_share_spec_from_yaml = YAML.load(yaml_str)
    except NameError:
        print(yaml_str)
        print(repr(test_share_spec.__class__))
        print(repr(YAML.representer.yaml_representers[test_share_spec.__class__]))
        raise

    if test_share_spec_from_yaml != test_share_spec:
        print(repr(test_share_spec))
        print(repr(test_share_spec_from_yaml))
        raise AssertionError(
            "MarketShareSpecs are not equal: {_test_instance_from_yaml!r}\n!=\n{_test_instance!r}"
        )
