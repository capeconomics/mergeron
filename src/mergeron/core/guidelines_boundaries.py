"""
Methods for defining and analyzing boundaries for Guidelines standards.

Includes function to create a canvas on which to draw boundaries for
Guidelines standards.

"""

from __future__ import annotations

import decimal
from typing import Literal
from typing import get_args

import numpy as np
from attrs import Attribute
from attrs import field
from attrs import frozen
from attrs import validators
from mpmath import mp

from .. import DEFAULT_REC
from .. import VERSION
from .. import YAML
from .. import ArrayDouble
from .. import RECForm
from .. import UPPAggregator
from .. import yamlize_attrs
from . import MGThresholds
from . import PubYear
from . import guidelines_boundary_functions as gbfn

__version__ = VERSION


mp.dps = 32
mp.trap_complex = True


@YAML.register_class
@frozen
class GuidelinesStandards:
    """
    Guidelines standards by Guidelines publication year.

    ΔHHI, Recapture Rate, GUPPI, and Diversion ratio thresholds
    constructed from concentration standards in Guidelines published in
    1992, 2010, and 2023. CMCR and IPR thresholds are drawn from literature.
    Note that the 2023 standards do not include a negative presumption
    or safeharbor, and the imputed presumption is one
    supplied here --- imputed from the (numbers equivalent of) the
    post-merger HHI threshold specified in the Guidelines presumption of harm.
    """

    pub_year: PubYear = field(kw_only=False, default=2023)
    """Year of publication of the Guidelines."""

    @pub_year.validator
    def _pub_year(self, _: Attribute[PubYear], _v: PubYear) -> None:
        if _v not in (_p := get_args(PubYear)):
            raise ValueError(
                f"Value given, {_v} is not a valid Guidelines publication year."
                f"Must be one of, {_p}."
            )

    safeharbor: MGThresholds = field(kw_only=True, default=None, init=False)
    """
    Negative presumption quantified on various measures.

    ΔHHI safeharbor bound, default recapture rate, GUPPI bound,
    diversion ratio limit, CMCR, and IPR.
    """

    presumption: MGThresholds = field(kw_only=True, default=None, init=False)
    """
    Presumption of harm specified in Guidelines.

    ΔHHI bound and corresponding default recapture rate, GUPPI bound,
    diversion ratio limit, CMCR, and IPR.
    """

    imputed_presumption: MGThresholds = field(kw_only=True, default=None, init=False)
    """
    Presumption of harm imputed from Guidelines.

    ΔHHI bound inferred from strict numbers-equivalent
    of (post-merger) HHI presumption, and corresponding default recapture rate,
    GUPPI bound, diversion ratio limit, CMCR, and IPR.
    """

    def __attrs_post_init__(self, /) -> None:
        """
        Initialize Guidelines thresholds, based on Guidelines publication year.

        In the 2023 Guidelines, the agencies do not define a
        negative presumption, or safeharbor. Practically speaking,
        given resource constraints and loss aversion, it is likely
        that staff only investigates mergers that meet the presumption;
        thus, here, the tentative delta safeharbor under
        the 2023 Guidelines is 100 points.
        """
        hhi_p, dh_s, dh_p = {
            1992: (0.18, 0.005, 0.01),
            2010: (0.25, 0.01, 0.02),
            2023: (0.18, 0.01, 0.01),
        }[self.pub_year]

        object.__setattr__(
            self,
            "safeharbor",
            MGThresholds(
                dh_s,
                _fc := int(np.ceil(1 / hhi_p)),
                _r := gbfn.round_cust(_fc / (_fc + 1), frac=0.05),
                _g := guppi_from_delta(dh_s, m_star=1.0, r_bar=_r),
                _dr := 1 - _r,
                _cmcr := _g,  # Not strictly a Guidelines standard
                _ipr := _g,  # Not strictly a Guidelines standard
            ),
        )

        object.__setattr__(
            self, "presumption", MGThresholds(dh_p, _fc, _r, _g, _dr, _cmcr, _ipr)
        )

        # imputed_presumption is relevant for presumptions implicating
        # mergers *to* symmetry in numbers-equivalent of post-merger HHI,
        # as in 2010 U.S.Guidelines.
        object.__setattr__(
            self,
            "imputed_presumption",
            (
                MGThresholds(
                    2 * (0.5 / _fc) ** 2,
                    _fc,
                    _r_i := gbfn.round_cust((_fc - 1 / 2) / (_fc + 1 / 2), frac=0.05),
                    _g,
                    (1 - _r_i) / 2,
                    _cmcr,
                    _ipr,
                )
                if self.pub_year == 2010
                else MGThresholds(
                    2 * (1 / (_fc + 1)) ** 2, _fc, _r, _g, _dr, _cmcr, _ipr
                )
            ),
        )


@frozen
class ConcentrationBoundary:
    """Concentration parameters, boundary coordinates, and area under concentration boundary."""

    threshold: float = field(kw_only=False, default=0.01)

    @threshold.validator
    def _tv(self, _: Attribute[float], _v: float, /) -> None:
        if not 0 <= _v <= 1:
            raise ValueError("Concentration threshold must lie between 0 and 1.")

    measure_name: Literal[
        "ΔHHI",
        "Combined share",
        "HHI contribution, pre-merger",
        "HHI contribution, post-merger",
    ] = field(kw_only=False, default="ΔHHI")

    @measure_name.validator
    def _mnv(self, _: Attribute[str], _v: str, /) -> None:
        if _v not in {
            "ΔHHI",
            "Combined share",
            "HHI contribution, pre-merger",
            "HHI contribution, post-merger",
        }:
            raise ValueError(f"Invalid name for a concentration measure, {_v!r}.")

    precision: int = field(
        kw_only=True, default=5, validator=validators.instance_of(int)
    )

    area: float = field(init=False, kw_only=True)
    """Area under the concentration boundary."""

    coordinates: ArrayDouble = field(init=False, kw_only=True, converter=ArrayDouble)
    """Market-share pairs as Cartesian coordinates of points on the concentration boundary."""

    def __attrs_post_init__(self, /) -> None:
        """Initialize boundary and area based on other attributes."""
        match self.measure_name:
            case "ΔHHI":
                conc_fn = gbfn.hhi_delta_boundary
            case "Combined share":
                conc_fn = gbfn.combined_share_boundary
            case "HHI contribution, pre-merger":
                conc_fn = gbfn.hhi_pre_contrib_boundary
            case "HHI contribution, post-merger":
                conc_fn = gbfn.hhi_post_contrib_boundary

        boundary_ = conc_fn(self.threshold, dps=self.precision)
        object.__setattr__(self, "area", boundary_.area)
        object.__setattr__(self, "coordinates", boundary_.coordinates)


@frozen
class DiversionBoundary:
    R"""
    Diversion ratio specification, boundary coordinates, and area under boundary.

    Along with the default diversion ratio and recapture rate,
    a diversion ratio boundary specification includes the recapture form --
    whether fixed for both merging firms' products ("fixed") or
    consistent with share-proportionality, i.e., "inside-out";
    the method of aggregating diversion ratios for the two products, and
    the precision for the estimate of area under the diversion ratio boundary
    (also defines the number of points on the boundary).

    The GUPPI boundary is a continuum of conditional diversion ratio boundaries,

    .. math::

        d_{ij} \vert_{p_i, p_j, m_j} \triangleq \frac{g_i p_i}{m_j p_j} = \overline{d}

    with :math:`d_{ij}` the diversion ratio from product :math:`i` to product :math:`j`;
    :math:`g_i` the GUPPI for product :math:`i`;
    :math:`m_j` the price-cost margin on product :math:`j`;
    :math:`p_i, p_j` the prices of goods :math:`i, j`, respectively; and
    :math:`\overline{d}` the diversion ratio threshold (i.e., bound).

    """

    diversion_ratio: float = field(kw_only=False, default=0.065)

    @diversion_ratio.validator
    def _dvv(self, _: Attribute[float], _v: float, /) -> None:
        if not (isinstance(_v, decimal.Decimal | float) and 0 <= _v <= 1):
            raise ValueError(
                "Margin-adjusted benchmark diversion share must lie between 0 and 1."
            )

    recapture_rate: float = field(
        kw_only=False, default=DEFAULT_REC, validator=validators.instance_of(float)
    )

    recapture_form: RECForm | None = field(kw_only=True, default=None)
    R"""
    The form of the recapture rate calculation.

    When :attr:`mergeron.RECForm.INOUT`, the recapture rate for
    he product having the smaller market-share is assumed to equal the given
    recapture rate and the recapture rate for the product with
    the larger market-share is computed assuming MNL demand.
    The given recapture rate can be estimated from the numbers-equivalent
    for the HHI threshold in the Guidelines presumption, or estimated from
    generated purchase probabilities.

    Fixed recapture rates are specified as :attr:`mergeron.RECForm.FIXED`. Fixed
    recapture rates are inconsistent with theory except under symmetric
    demand and prices, but may be relevant for comparison to external results.

    The default is :attr:`mergeron.RECForm.INOUT` except that it is `None` when
    the aggregation method is :attr:`mergeron.UPPAggregator.MAX`.
    """

    @recapture_form.validator
    def _rsv(self, _: Attribute[RECForm | None], _v: RECForm | None, /) -> None:
        if _v is not None and not (isinstance(_v, RECForm)):
            raise ValueError(f"Invalid recapture specification, {_v!r}.")

    aggregator: UPPAggregator = field(
        kw_only=True,
        default=UPPAggregator.MIN,
        validator=validators.instance_of(UPPAggregator),
    )
    """
    Method for aggregating the distinct diversion ratio measures for the two products.

    Distinct diversion ratio or GUPPI measures for the two merging-firms' products are
    aggregated using the method specified here, using :class:`mergeron.UPPAggregator`.
    """

    precision: int = field(
        kw_only=True, default=5, validator=validators.instance_of(int)
    )
    """
    The number of decimal places of precision for the estimated area under the UPP boundary.

    Leaving this attribute unspecified will result in the default precision,
    which varies based on the `aggregator` attribute, reflecting
    the limit of precision available from the underlying functions. The number of
    boundary points generated is also defined based on this attribute.

    """

    area: float = field(init=False, kw_only=True)
    """Area under the diversion ratio boundary."""

    coordinates: ArrayDouble = field(init=False, kw_only=True, converter=ArrayDouble)
    """Market-share pairs as Cartesian coordinates of points on the diversion ratio boundary."""

    def __attrs_post_init__(self, /) -> None:
        """Initialize boundary and area based on other attributes."""
        _diversion_share = critical_diversion_share(
            self.diversion_ratio, r_bar=self.recapture_rate
        )

        upp_agg_kwargs: gbfn.DiversionShareBoundaryKeywords = {"dps": self.precision}

        if self.aggregator == UPPAggregator.MAX:
            print(
                "NOTE: Recapture form has no effect when the aggregation method is "
                ":attr:`mergeron.UPPAggregator.MAX`. Ignored here."
            )
        elif self.recapture_form == RECForm.OUTIN:
            raise NotImplementedError(
                "Diversion-share boundary construction for `mergeron.RECForm.OUTIN` is not implemented. "
                "Please use `mergeron.RECForm.INOUT` and specify the recapture rate of "
                "the smaller merging-firm's product as `recapture_rate`."
            )
        elif self.recapture_form is None:
            object.__setattr__(self, "recapture_form", RECForm.INOUT)
            upp_agg_kwargs |= {"recapture_form": "inside-out"}
        else:
            upp_agg_kwargs |= {"recapture_form": self.recapture_form.value}

        match self.aggregator:
            case UPPAggregator.AVG:
                upp_agg_fn = gbfn.diversion_share_boundary_xact_avg
            case UPPAggregator.MAX:
                upp_agg_fn = gbfn.diversion_share_boundary_max
                upp_agg_kwargs |= {"dps": 10}
            case UPPAggregator.MIN:
                upp_agg_fn = gbfn.diversion_share_boundary_min
                upp_agg_kwargs |= {"dps": 10}
            case UPPAggregator.DIS:
                upp_agg_fn = gbfn.diversion_share_boundary_wtd_avg
                upp_agg_kwargs |= {"aggregator": "distance", "weighting": None}
            case _:
                upp_agg_fn = gbfn.diversion_share_boundary_wtd_avg

                if self.aggregator.value.endswith("geometric mean"):
                    aggregator_ = "geometric mean"
                elif self.aggregator.value.endswith("average"):
                    aggregator_ = "arithmetic mean"
                else:
                    aggregator_ = "distance"

                if self.aggregator.value.startswith("cross-product-share"):
                    weighting_ = "cross-product-share"
                elif self.aggregator.value.startswith("own-share"):
                    weighting_ = "own-share"
                else:
                    weighting_ = None

                upp_agg_kwargs |= {"aggregator": aggregator_, "weighting": weighting_}  # type: ignore[typeddict-item]

        _boundary = upp_agg_fn(_diversion_share, self.recapture_rate, **upp_agg_kwargs)  # type: ignore[misc]
        object.__setattr__(self, "area", _boundary.area)
        object.__setattr__(self, "coordinates", _boundary.coordinates)


def guppi_from_delta(
    _delta_bound: float = 0.01, /, *, m_star: float = 1.00, r_bar: float = DEFAULT_REC
) -> float:
    """
    Translate ∆HHI bound to GUPPI bound.

    Parameters
    ----------
    _delta_bound
        Specified ∆HHI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
        GUPPI bound corresponding to ∆HHI bound, at given margin and recapture rate.

    """
    return gbfn.round_cust(
        m_star * r_bar * (_s_m := np.sqrt(_delta_bound / 2)) / (1 - _s_m), frac=0.005
    )


def critical_diversion_share(
    _guppi_bound: float = 0.075,
    /,
    *,
    m_star: float = 1.00,
    r_bar: float = 1.00,
    frac: float = 1e-16,
) -> float:
    """
    Corollary to GUPPI bound.

    Parameters
    ----------
    _guppi_bound
        Specified GUPPI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
        Critical diversion share (diversion share bound) corresponding to the GUPPI bound
        for given margin and recapture rate.

    """
    return gbfn.round_cust(_guppi_bound / (m_star * r_bar), frac=frac)


def share_from_guppi(
    _guppi_bound: float = 0.065, /, *, m_star: float = 1.00, r_bar: float = DEFAULT_REC
) -> float:
    """
    Symmetric-firm share for given GUPPI, margin, and recapture rate.

    Parameters
    ----------
    _guppi_bound
        GUPPI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
    float
        Symmetric firm market share on GUPPI boundary, for given margin and
        recapture rate.

    """
    return gbfn.round_cust(
        (_d0 := critical_diversion_share(_guppi_bound, m_star=m_star, r_bar=r_bar))
        / (1 + _d0)
    )


if __name__ == "__main__":
    print(
        "This module defines classes with methods for generating boundaries for concentration and diversion-ratio screens."
    )


for _typ in (
    ConcentrationBoundary,
    DiversionBoundary,
    GuidelinesStandards,
    MGThresholds,
):
    yamlize_attrs(_typ)
