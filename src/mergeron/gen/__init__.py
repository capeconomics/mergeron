"""Defines constants, specifications and containers for industry data generation and testing."""

from __future__ import annotations

import enum
import io
import os
from collections.abc import Sequence
from operator import attrgetter
from typing import IO
from typing import TYPE_CHECKING

import h5py
import hdf5plugin
import numpy as np
from attrs import Attribute
from attrs import Converter
from attrs import cmp_using
from attrs import field
from attrs import frozen

if TYPE_CHECKING:
    from numpy.random import SeedSequence

    from .. import zipfile

from .. import DEFAULT_REC
from .. import EMPTY_ARRAYDOUBLE
from .. import EMPTY_ARRAYUINT8 as EMPTY_ARRAYUINT8
from .. import VERSION
from .. import YAML
from .. import ArrayBIGINT
from .. import ArrayBoolean as ArrayBoolean
from .. import ArrayDouble
from .. import ArrayFloat
from .. import ArrayINT
from .. import ArrayUINT8 as ArrayUINT8
from .. import Enameled
from .. import RECForm
from .. import UPPAggregator
from .. import yamlize_attrs
from ..core import DEFAULT_BETA_DIST_PARMS
from ..core import DEFAULT_DIST_PARMS
from ..core import EmpiricalMarginData
from ..core.empirical_margin_distribution import margin_data_builder

__version__ = VERSION

SUBSAMPLE_SIZE = int(os.getenv("MERGERON_SUBSAMPLE_SIZE", "100_000"))
"""
Subsample size for parallelization.

Can be specified by setting the environment variable, `MERGERON_SUBSAMPLE_SIZE`.
"""

# TODO: Tailor the parameters to the size-distribution of U.S. firms from
# the Census Data product, EC2200SIZEREVFIRM - "Selected Sectors:
# Sales, Value of Shipments, or Revenue Size of Firms for the U.S.: 2022"
# See, https://www.census.gov/data/tables/2022/econ/economic-census/naics-sector-00.html
HSR_BETA_PARMS = tuple(
    float(_s) for _s in os.getenv("MERGERON_HSR_BETA_PARMS", "1.5, 20").split(", ")
)
"""
Beta parameters for the version of the HSR filing test specified
by :attr:`HSRFilingTest.SoP_RND`.

Can be specified by setting the environment variable, `MERGERON_HSR_BETA_PARMS`.
"""

DEFAULT_FCOUNT_WTS: ArrayFloat = ArrayFloat((_nr := np.arange(6, 0, -1)) / _nr.sum())
DEFAULT_DIR_COND_DIST_PARMS = ArrayFloat([2.5, 2.5, 1.0])
HSR_RATIO = 10
r"""
Ratio of large-firm to small-firm revenues, from HSR size-of-person test.

See, https://www.ftc.gov/enforcement/premerger-notification-program/.

The HSR filing thresholds, of $10 million and $100 million (in 1976 dollars) for
the size-of-person test ("SOP" test) are defined in part (a) Filing of 15 U.S.C. 18a.
Premerger notification and waiting period, under (2)(B)(ii)(I). The HSR filing thresholds
are subject to annual revisions at the rate of inflation, but the 10-to-1 revenue ratio
of larger merging firm revenues to smaller merging firm revenues prevails.

A simple form of the HSR filing test would impose a 10-to-1
ratio restriction (max.-to-min.) on the merging firms' revenues. This
version of the filing test is implemented by specifying `hsr_filing_test_type`
as :attr:`HSRFilingTest.SoP_TEN`.

Note, however, the filing test does not *require* that
the larger merging firm be ten times the size of the smaller merging firm.
If both firms are of approximately the same size and each has revenues
greater than $100 million, adjusted for inflation, the proposed merger meets
the HSR filing test despite the ratio of the merging firm's revenues being
1-to-1.

The mere fact that one merging firm's revenues are no less than ten times the others'
is also not sufficient, in itself, for meeting the HSR size-of-person filing test ---
for example, a proposed merger does not require a filing if the larger firm's revenue is
$101 million, adjusted for inflation, and the smaller firm's revenue is $9 million, adjusted
for inflation, and none of the other filing requirements are met, despite the
ratio of merging firms' revenues being greater than 10-to-1.

Accordingly, this package supports an alternate assumption regarding the filing test,
as follows: supposed that a firm with share :math:`\mathcal{s}` has revenues, :math:`\mathcal{R}`,
equal to $10 million, adjusted for inflation, at the *n*-th firm's price. The HSR SOP test is met
if the smaller merging firm's revenues are no less than :math:`\mathcal{R}` and
the larger merging firm's revenues are no less than 10 times :math:`\mathcal{R}`.
The enumeration, :attr:`HSRFilingTest.SoP_RND`, denotes this version of the filing test with
randomly-generated :math:`\mathcal{s}`, while the enumeration :attr:`HSRFilingTest.SoP_NTH` denotes
this version of the filing test but with the *n*-th firm's share taken as :math:`\mathcal{s}`.

The parameters of the Beta distribution for generating random :math:`\mathcal{s}` are
:math:`\alpha = 1.5` and :math:`\beta = 2.0`. These parameters can be changed
by setting the environment variable, `MERGERON_HSR_BETA_PARMS` to a comma-separated list
of values for :math:`\alpha` and :math:`\beta`.

"""


@frozen
class HMTSpec:
    """Market definition test parameters, with defaults.

    Attributes specify
    1) the (index of the) product for which to test
    the existence of a relevant antitrust market---0 indicates
    the first merging firm's product and 1 indicates the second
    merging firm's product,
    2) the level of the SSNIP---5% by default,
    3) the passthrough rate---50% by default, and
    4) whether to recompute shares as market shares in
    the relevant antitrust market for the given merging
    firm.

    """

    product_index: int = 0
    ssnip_level: float = 0.05
    passthrough_rate: float = 0.5
    recompute_shares: bool = False


DEFAULT_HMT_FLAG = HMTSpec()


@frozen
class SeedSequenceData:
    """Seed sequence values for shares, margins, and, optionally, firm-counts and prices."""

    share: SeedSequence = field(eq=attrgetter("state"))
    pcm: SeedSequence = field(eq=attrgetter("state"))
    fcounts: SeedSequence | None = field(eq=lambda x: x if x is None else x.state)
    price: SeedSequence | None = field(
        default=None, eq=lambda x: x if x is None else x.state
    )
    hsr_filing_test: SeedSequence | None = field(
        default=None, eq=lambda x: x if x is None else x.state
    )


@YAML.register_class
@enum.unique
class PriceSpec(str, Enameled):
    """Price specification.

    The structure of prices, whether symmetric, randomly distributed, or
    share_correlated, whether directly or derived from the cost-structure
    and generated price-cost margins.
    """

    PRICE_SYM = "price-symmetry"
    """Prices normalized to unity (1.0)."""

    PRICE_RND = "price, random, in steps"
    """Prices randomly drawn from, the set {0.2, 0.4, 0.6, 0.8, 1.0}."""

    PRICE_POS = "price, positive share-correlation"
    """Prices ranging from from 0.2 to 1.0, by 0.2, with increasing share.

    For example, firms having share 20% or less have prices of 0.2, firms having
    share 40% or less but greater than 20% have prices of 0.4, and so on, with firms
    having share greater than 80% having prices of 1.

    """

    PRICE_NEG = "price, negative share-correlation"
    """Prices ranging from 1.0 to 0.2, by 0.2, with increasing share."""

    COST_SYM = "cost-symmetry"
    """Costs normalized to unity (1.0)."""

    COST_RND = "cost, random, in steps"
    """Costs drawn from, the set {0.2, 0.4, 0.6, 0.8, 1.0}."""

    COST_POS = "cost, positive share-correlation"
    """Costs ranging from from 0.2 to 1.0, by 0.2, with increasing share."""

    COST_NEG = "cost, negative share-correlation"
    """Costs ranging from 1.0 to 0.2, by 0.2, with increasing share."""


@YAML.register_class
@enum.unique
class SHRDistribution(str, Enameled):
    """Market share distributions."""

    UNI = "Uniform"
    R"""Uniform distribution over :math:`s_1 + s_2 \leqslant 1`"""

    DIR_FLAT = "Flat Dirichlet"
    """Shape parameter for all merging-firm-shares is unity (1)"""

    DIR_FLAT_CONSTR = "Flat Dirichlet - Constrained"
    """Impose minimum probability weight on each firm-count

    Only firm-counts with probability weight of 3% or more
    are included for data generation.
    """

    DIR_ASYM = "Asymmetric Dirichlet"
    """Share distribution for merging-firm shares has a higher peak share

    By default, shape parameter for merging-firm-share is 2.5, and
    1.0 for all others. Defining, :attr:`.ShareSpec.parameters`
    as a vector of shape parameters with length matching
    that of :attr:`.ShareSpec.parameters` allows flexible specification
    of Dirichlet-distributed share-data generation.
    """

    DIR_COND = "Conditional Dirichlet"
    """Shape parameters for non-merging firms is proportional

    Shape parameters for merging-firm-share are given as the first two
    elements of :attr:`.ShareSpec.parameters`; the shape parameters
    for the remaining firms' shares are equiproportional and sum to the
    third element of :attr:`.ShareSpec.parameters`. See,
    Balakrishnan [#balakrishnan2006]_ (p. 25).

    .. [#balakrishnan2006] Balakrishnan, S. (2006). Continuous Multivariate Distributions.
      *In* John Wiley & Sons, Wiley StatsRef: Statistics Reference Online (2014).
    """


def _fcounts_weights_conv(
    _v: Sequence[float | int] | ArrayDouble | ArrayINT | None, _i: ShareSpec
) -> ArrayFloat | None:
    if _i.distribution == SHRDistribution.UNI:
        return None
    elif _v is None or len(_v) == 0 or np.array_equal(_v, DEFAULT_FCOUNT_WTS):
        return DEFAULT_FCOUNT_WTS
    else:
        return ArrayFloat(
            _tv if (_tv := np.asarray(_v, float)).sum() == 1 else _tv / _tv.sum()
        )


def _shr_dp_conv(_v: Sequence[float] | ArrayFloat | None, _i: ShareSpec) -> ArrayFloat:
    retval = np.array([], float)
    if _v is None or len(_v) == 0 or np.array_equal(_v, DEFAULT_DIST_PARMS):
        if _i.distribution == SHRDistribution.UNI:
            return DEFAULT_DIST_PARMS
        elif _i.distribution == SHRDistribution.DIR_COND:
            # alpha-value for merging firms; alpha for non-merging firms
            # computed as in Balakrishnan, 2005, p. 25
            return DEFAULT_DIR_COND_DIST_PARMS
        else:
            fc_max = 1 + len(_i.firm_counts_weights)  # type: ignore[arg-type]

            match _i.distribution:
                case SHRDistribution.DIR_FLAT | SHRDistribution.DIR_FLAT_CONSTR:
                    retval = np.ones(fc_max)
                case SHRDistribution.DIR_ASYM:
                    retval = np.array([2.0] * 6 + [1.5] * 5 + [1.25] * fc_max)
                case _ if isinstance(_i.distribution, SHRDistribution):
                    raise ValueError(
                        f"No default defined for market share distribution, {_i.distribution!r}"
                    )
                case _:
                    raise ValueError(
                        f"Unsupported distribution for market share generation, {_i.distribution!r}"
                    )
    elif isinstance(_v, Sequence | np.ndarray):
        retval = np.asarray(_v, float)
    else:
        raise ValueError(
            f"Input, {_v!r} has invalid type. Must be None, Sequence of floats, or Numpy ndarray."
        )

    return ArrayFloat(retval)


@frozen
class ShareSpec:
    """Market share specification.

    Choosing a Dirichlet-type distribution allows pooling data generated for
    markets with varying numbers of firms, specifying by a vector of firm-count
    weights with the first element giving the relative proportion of two-firm markets
    and on down the line. Choosing uniformly distributed shares allows generation of
    data for markets with unspecified numbers of firms.

    Notes
    -----
    If :attr:`.distribution` == :attr:`.SHRDistribution.UNI`, it is then infeasible that
    :attr:`.recapture_form` == :attr:`mergeron.RECForm.OUTIN`.
    In other words, recapture rates cannot be estimated using
    outside-good choice probabilities if the distribution of markets over firm-counts
    is unspecified.

    We impose the further restriction that if shares have a Dirichlet distribution,
    then recapture rates cannot be fixed (:attr:`mergeron.RECForm.FIXED`). As the
    generated data are sufficient for computing exact recapture rates for each firm and
    exact recapture rates are not fixed, the above restriction enforces internal
    consistency.

    Fixed recapture rates are inconsistent with theory, unless demand and prices are symmetric,
    but are supported here to allow comparison with external results.

    """

    distribution: SHRDistribution = field(kw_only=False)
    """See :class:`SHRDistribution`"""

    firm_counts_weights: ArrayFloat | None = field(
        kw_only=True,
        eq=cmp_using(eq=np.array_equal),
        converter=Converter(_fcounts_weights_conv, takes_self=True),  # type: ignore
    )
    """Relative or absolute frequencies of pre-merger firm counts

    Defaults to :attr:`DEFAULT_FCOUNT_WTS`, which specifies pre-merger
    firm-counts of 2 to 7 with weights in descending order from 6 to 1.

    ALERT: Firm-count weights are irrelevant when the merging firms' shares are specified
    to have uniform distribution; therefore this attribute is forced to None if
    :attr:`.distribution` == :attr:`.SHRDistribution.UNI`.
    """

    @firm_counts_weights.default
    def _fcwd(self) -> ArrayFloat | None:
        return _fcounts_weights_conv(None, self)

    @firm_counts_weights.validator
    def _fcwv(self, _: Attribute[ArrayFloat], _v: ArrayFloat) -> None:
        if self.distribution != SHRDistribution.UNI and not _v.size:
            raise ValueError(
                f"Attribute, {'"firm_counts_weights"'} must be populated if the share distribution is "
                "other than uniform distribution."
            )

    parameters: ArrayFloat | ArrayDouble = field(
        kw_only=True,
        converter=Converter(_shr_dp_conv, takes_self=True),  # type: ignore
        eq=cmp_using(eq=np.array_equal),
    )
    """Parameters for tailoring market-share distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Dirichlet-type distributions, a vector of shape parameters of length
    equal to 1 plus the length of firm-count weights below; defaults depend on
    type of Dirichlet-distribution specified.
    """

    @parameters.default
    def _dpd(self) -> ArrayFloat:
        # converters run after defaults, and we
        # avoid redundancy and confusion here
        return _shr_dp_conv(None, self)

    lower_bound: float = field(kw_only=True, default=0.0)
    """Restriction on market share draws.

    Minimum market share in each draw must be greater than this value (i.e., the
    lower bound in non-inclusive, or "open"). Defaults to 0.

    Note that the upper bound share for each firm is automatically 1 - lower_bound times
    the number of firms, which limits the feasible range for the lower_bound.

    This threshold is not enforced when shares are drawn from the
    conditional Dirichlet distribution [#balakrishnan]_ specified in this package,
    in which the alpha parameters for generating the merging firms shares
    are [2.5, 2.5, 1.0], and the alpha parameters for generating
    the non-merging firms's shares are computed to sum to 1.0 and replace
    the final entry in the vector given above. The conditional Dirichlet distribution
    as implemented here is useful for maintaining that the distribution of
    merging-firm shares is not decreasing in the number of firms in the market, as with
    share generated using other forms of the Dirichlet distribution.

    .. [#balakrishnan] Balakrishnan, S. (2006). Continuous Multivariate Distributions.
      *In* John Wiley & Sons, Wiley StatsRef: Statistics Reference Online (2014).
    """

    recapture_form: RECForm = field(default=RECForm.INOUT)
    """See :class:`mergeron.RECForm`"""

    @recapture_form.validator
    def _rfv(self, _: Attribute[RECForm], _v: RECForm) -> None:
        if self.distribution == SHRDistribution.UNI and _v == RECForm.OUTIN:
            raise ValueError(
                "Outside-good choice probabilities cannot be generated if the "
                "merging firms' market shares have uniform distribution over the "
                "3-dimensional simplex --- the shares of non-merging firms are "
                "indeterminate."
            )
        elif "DIR" in self.distribution.name and _v == RECForm.FIXED:
            raise ValueError(
                "Recapture rates cannot be held fixed for all firms when "
                "shares and diversion ratios of non-merging firms are drawn from "
                "Dirichlet distributions and, therefore, are known."
            )

    recapture_rate: float | None = field(kw_only=True)
    """A value between 0 and 1.

    :code:`None` if market share specification requires direct generation of
    outside good choice probabilities (:attr:`mergeron.RECForm.OUTIN`).

    The recapture rate is usually calibrated to the numbers-equivalent of the
    HHI threshold for the presumption of harm from unilateral competitive effects
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
    def _rrd(self) -> float | None:
        return None if self.recapture_form == RECForm.OUTIN else DEFAULT_REC

    @recapture_rate.validator
    def _rrv(self, _: Attribute[float], _v: float) -> None:
        if _v and not (0 < _v <= 1):
            raise ValueError("Recapture rate must lie in the interval, [0, 1).")

    def __post_init__(self) -> None:
        """Validate restrictions on multiple attributes here."""
        if (
            self.firm_counts_weights is not None
            and len(self.parameters) < (1 + len(self.firm_counts_weights))
            and self.distribution != SHRDistribution.DIR_COND
        ):
            print(self)
            raise ValueError(
                "If specified, the number of distribution parameters must equal or "
                "exceed the maximum firm-count premerger, namely 1 plus"
                "the length of the vector specifying firm-count weights."
            )
        elif self.distribution == SHRDistribution.DIR_COND and len(
            self.parameters
        ) != len(("theta_1", "theta_2", "Theta")):
            raise ValueError(
                f"Given number, {len(self.parameters)} of parameters "
                f'for PCM with distribution, "{self.distribution}" is incorrect.'
            )


@YAML.register_class
@enum.unique
class PCMDistribution(str, Enameled):
    """Margin distributions."""

    UNI = "Uniform"
    BETA = "Beta"
    EMPR_U = "Damodaran margin data, uni-modal resampling"
    EMPR_M = "Damodaran margin data, multi-modal resampling"


@YAML.register_class
@enum.unique
class PCMRestriction(str, Enameled):
    """Restriction on generated Firm 2 margins."""

    IID = "independent and identically distributed (IID)"
    MNL = "Nash-Bertrand equilibrium with multinomial logit (MNL) demand"
    SYM = "symmetric"


def _pcm_parms_conv(
    _v: ArrayFloat | Sequence[float] | None, _i: PCMSpec
) -> ArrayFloat | ArrayDouble | EmpiricalMarginData:
    if _i.distribution in {PCMDistribution.EMPR_U, PCMDistribution.EMPR_M} and (
        _v is None or isinstance(_v, EmpiricalMarginData)
    ):
        return _v or margin_data_builder()
    elif _v is None or len(_v) == 0 or np.array_equal(_v, DEFAULT_DIST_PARMS):
        match _i.distribution:
            case PCMDistribution.BETA:
                return DEFAULT_BETA_DIST_PARMS
            case _:
                return DEFAULT_DIST_PARMS
    elif isinstance(_v, Sequence | np.ndarray):
        return ArrayFloat(np.array(_v, float) if isinstance(_v, Sequence) else _v)
    else:
        raise ValueError(
            f"Input, {_v!r} has invalid type. Must be None, sequence of floats,"
            "sequence of Numpy arrays, or Numpy ndarray."
        )


@frozen
class PCMSpec:
    """Price-cost margin (PCM) specification.

    If price-cost margins are specified as having Beta distribution,
    `parameters` is specified as a pair of positive, non-zero shape parameters of
    the standard Beta distribution. Specifying shape parameters :code:`np.array([1, 1])`
    is known equivalent to specifying uniform distribution over
    the interval :math:`[0, 1]`. If price-cost margins are specified as having
    Bounded-Beta distribution, `parameters` is specified as
    the tuple, (`mean`, `std deviation`, `min`, `max`), where `min` and `max`
    are lower- and upper-bounds respectively within the interval :math:`[0, 1]`.

    """

    distribution: PCMDistribution = field(default=PCMDistribution.UNI)
    """See :class:`PCMDistribution`"""

    parameters: ArrayFloat | ArrayDouble | EmpiricalMarginData = field(
        kw_only=True,
        eq=cmp_using(
            lambda _i, _j: (
                _i == _j
                if isinstance(_i, EmpiricalMarginData)
                else np.array_equal(_i, _j)
            )
        ),
        converter=Converter(_pcm_parms_conv, takes_self=True),  # type: ignore
    )
    """Parameter specification for tailoring PCM distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Beta distribution, shape parameters, defaults to `(1, 1)`;
    for Bounded-Beta distribution, vector of (min, max, mean, std. deviation), non-optional;
    for Empirical distribution, the converter functions supplies a computed data structure
    """

    @parameters.default
    def _dpwd(self) -> ArrayDouble | ArrayFloat | EmpiricalMarginData:
        return _pcm_parms_conv(None, self)

    @parameters.validator
    def _dpv(
        self,
        _: Attribute[ArrayFloat | Sequence[ArrayDouble] | None],
        _v: ArrayFloat | Sequence[ArrayDouble] | None,
    ) -> None:
        if self.distribution.name.startswith("BETA"):
            if (
                _v is None
                or not hasattr(_v, "len")
                or (isinstance(_v, np.ndarray) and not any(_v.shape))
            ):
                pass
            elif np.array_equal(_v, DEFAULT_DIST_PARMS):
                raise ValueError(
                    f"The distribution parameters, {DEFAULT_DIST_PARMS!r} "
                    "are not valid with margin distribution, {_dist_type_pcm!r}"
                )
            elif (
                self.distribution == PCMDistribution.BETA and len(_v) != len(("a", "b"))
            ) or (
                self.distribution == PCMDistribution.EMPR_U
                and len(_v) != len(("mu", "sigma", "max", "min"))
            ):
                raise ValueError(
                    f"Given number, {len(_v)} of parameters "
                    f'for PCM with distribution, "{self.distribution}" is incorrect.'
                )

        elif self.distribution == PCMDistribution.EMPR_M and not isinstance(
            _v, EmpiricalMarginData
        ):
            raise ValueError(
                "Parameter input for empirical distribution must be structured as EmpiricalMarginData."
            )

    pcm_restriction: PCMRestriction = field(kw_only=True, default=PCMRestriction.IID)
    """See :class:`PCMRestriction`"""


@YAML.register_class
@enum.unique
class HSRFilingTest(str, Enameled):
    """
    Implementations of the HSR size-of-person test.

    See,
    https://www.ftc.gov/sites/default/files/attachments/premerger-introductory-guides/guide2.pdf,
    particularly, Sections II and V(c).

    """

    NONE = "Unrestricted"
    """A HSR filing is assumed for (the hypothetical transaction in) every draw.

    In effect, one assumes that, in every draw, either
    (a) the value of the hypothetical acquisition is $50 million or more, as adjusted, and
    the size-of-person test is met, or (b) the value of
    the hypothetical acquisition exceeds $200 million, as adjusted (and no exemptions apply).

    This is the default.
    """

    SoP_NTH = "n-th Firm meets lower threshold"
    """
    HSR size-of-person filing test against n-th firm share.

    When the :math:`n`-th firm's revenues in a market of :math:`n` firms are
    assumed to match the lower threshold for the size-of-person test,
    a filing is required if the smaller merging firm's revenue share is
    no less than the :math:`n`-th firm's revenue share, and
    the larger merging firm's revenue share is no less than 10 times the
    :math:`n`-th firm's revenue share. In effect, this version
    for the test assumes that at least one firm in every investigated market
    has revenues exactly equal to the dollar amount of the lower threshold for
    the HSR size-of-person test.
    """

    SoP_RND = "randomly-drawn test-firm-share meets lower threshold"
    """
    HSR size-of-person filing test against randomly-drawn test shares.

    In this version of the size-of-person test, a "test firm" share is drawn randomly
    from the Beta distribution with two parameters given by :attr:`HSR_BETA_PARMS`.
    If the smaller merging firm's revenue share is no less than the "test firm"'s
    revenue share, calculated at the :math:`n`-th firm's price, and the larger
    merging firm's revenue share is no less than 10 times the "test firm"'s
    revenue share, a filing is inferred to be required.
    """

    SoP_TEN = "smaller merging firm meets lower threshold"
    """
    HSR size-of-person filing test against smaller merging firm's share.

    When smaller merging-firm's revenues are assumed to match
    the lower threshold for the size-of-person test, a filing is required if
    the larger merging firm is at least 10 times as large as the smaller
    merging firm.
    """


@frozen(kw_only=True)
class MarketsData:
    """Container for generated market sample dataset."""

    shares: ArrayDouble = field(eq=cmp_using(eq=np.array_equal), converter=ArrayDouble)
    """Generated market shares, with zeros for markets with fewer firms than the maximum."""

    margins: ArrayDouble = field(eq=cmp_using(eq=np.array_equal), converter=ArrayDouble)
    """Generated prices; normalized to 1 in default specification)"""

    prices: ArrayDouble = field(eq=cmp_using(eq=np.array_equal), converter=ArrayDouble)
    """Generated price-cost margins (PCM)"""

    aggregate_choice_probability: ArrayDouble = field(
        eq=cmp_using(eq=np.array_equal),
        default=EMPTY_ARRAYDOUBLE,
        converter=ArrayDouble,
    )
    """
    One (1) minus probability that the outside good is chosen

    Converts market shares to choice probabilities by multiplication.
    """

    def to_h5bin(self) -> bytes:
        """Save market sample data to HDF5 file."""
        byte_stream = io.BytesIO()
        with h5py.File(byte_stream, "w") as _h5f:
            for _a in self.__attrs_attrs__:
                if all(((_arr := getattr(self, _a.name)).any(),)):
                    _h5f.create_dataset(
                        _a.name, data=_arr, fletcher32=True, **hdf5plugin.Zstd()
                    )
        return byte_stream.getvalue()

    @classmethod
    def from_h5f(
        cls, _hfh: io.BufferedReader | zipfile.ZipExtFile | IO[bytes]
    ) -> MarketsData:
        """Load market sample data from HDF5 file."""
        with h5py.File(_hfh, "r") as _h5f:
            _retval = cls(**{_a: _v[:] for _a, _v in _h5f.items()})
        return _retval


@YAML.register_class
@enum.unique
class INVResolution(str, Enameled):
    """Report investigations resulting in clearance; enforcement; or both, respectively."""

    CLRN = "clearance"
    ENFT = "enforcement"
    BOTH = "clearance and enforcement, respectively"


@frozen
class UPPTestRegime:
    """Configuration for UPP tests."""

    resolution: INVResolution = field(kw_only=False, default=INVResolution.ENFT)
    """Whether to test clearance, enforcement."""

    @resolution.validator
    def _resvdtr(
        _i: UPPTestRegime, _: Attribute[INVResolution], _v: INVResolution
    ) -> None:
        if _v == INVResolution.BOTH:
            raise ValueError(
                "GUPPI test cannot be performed with both resolutions; only useful for reporting"
            )
        elif _v not in {INVResolution.CLRN, INVResolution.ENFT}:
            raise ValueError(
                f"Must be one of, {INVResolution.CLRN!r} or {INVResolution.ENFT!r}"
            )

    guppi_aggregator: UPPAggregator = field(kw_only=False)
    """Aggregator for GUPPI test."""

    @guppi_aggregator.default
    def _gad(self) -> UPPAggregator:
        return (
            UPPAggregator.MIN
            if self.resolution == INVResolution.ENFT
            else UPPAggregator.MAX
        )

    diversion_aggregator: UPPAggregator = field(kw_only=False)
    """Aggregator for diversion ratio test."""

    @diversion_aggregator.default
    def _dad(self) -> UPPAggregator:
        return self.guppi_aggregator


@YAML.register_class
@enum.unique
class StatsGroup(str, Enameled):
    """Measure used to summarize investigations data."""

    FC = "ByFirmCount"
    DL = "ByDelta"
    HD = "ByHHIandDelta"
    ZN = "ByConcentrationZone"


@frozen
class UPPTestsCounts:
    """Counts of markets meeting a specified Guidelines standard.

    The specification includes Guidelines thresholds (:attr:`mergeron.core.MGThresholds`)
    as well as a test regime (:attr:`UPPTestRegime`).
    See, :method:`enforcement_counts.compute_enforcement_counts`().
    """

    ByFirmCount: ArrayBIGINT = field(
        eq=cmp_using(eq=np.array_equal), converter=ArrayBIGINT
    )
    ByDelta: ArrayBIGINT = field(eq=cmp_using(eq=np.array_equal), converter=ArrayBIGINT)
    ByHHIandDelta: ArrayBIGINT = field(
        eq=cmp_using(eq=np.array_equal), converter=ArrayBIGINT
    )


for _typ in (SeedSequenceData, ShareSpec, PCMSpec, UPPTestsCounts, UPPTestRegime):
    yamlize_attrs(_typ)
