"""Methods to generate data for analyzing merger enforcement policy."""

from __future__ import annotations

from itertools import starmap
from math import ceil
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from attrs import Attribute
from attrs import Converter
from attrs import define
from attrs import field
from attrs import validators
from joblib import Parallel
from joblib import delayed
from joblib import parallel_config
from numpy.random import SeedSequence

if TYPE_CHECKING:
    from ruamel import yaml

    from ..core import MGThresholds

from .. import EMPTY_ARRAYBIGINT
from .. import EMPTY_ARRAYBOOLEAN
from .. import NTHREADS
from .. import PKG_NAME
from .. import VERSION
from .. import YAML
from .. import ArrayBIGINT
from .. import ArrayBoolean
from .. import ArrayDouble
from .. import RECForm
from .. import yaml_rt_mapper
from .. import zipfile
from . import SUBSAMPLE_SIZE
from . import HMTSpec
from . import HSRFilingTest
from . import MarketsData
from . import PCMDistribution
from . import PCMRestriction
from . import PCMSpec
from . import PriceSpec
from . import SeedSequenceData
from . import ShareSpec
from . import SHRDistribution
from . import UPPTestRegime
from . import UPPTestsCounts
from .data_generation_functions import market_share_sampler
from .data_generation_functions import prices_sampler
from .enforcement_stats import StatsGroup
from .enforcement_stats import compute_enforcement_counts
from .upp_tests import compute_upp_test_counts

__version__ = VERSION


def _seed_data_conv(
    _v: SeedSequenceData
    | tuple[
        SeedSequence,
        SeedSequence,
        SeedSequence | None,
        SeedSequence | None,
        SeedSequence | None,
    ]
    | None,
    _i: MarketSample,
) -> SeedSequenceData:
    if isinstance(_v, SeedSequenceData):
        return _v

    shr_seed, pcm_seed = (
        _v[:2] if _v else tuple(SeedSequence(pool_size=8) for _ in range(2))
    )

    _uniform_share_distribution_flag = _i.share_spec.distribution == SHRDistribution.UNI
    fct_seed = (
        None
        if _i.share_spec.distribution == SHRDistribution.UNI
        else (_v[2] if _v else SeedSequence(pool_size=8))
    )

    _random_price_variation_flag = _i.price_spec.name.endswith("RND")
    if not _random_price_variation_flag:
        pri_seed = None
    elif not _v:
        pri_seed = SeedSequence(pool_size=8)
    else:
        pri_seed = _v[2] if _uniform_share_distribution_flag else _v[3]

    _hsr_test_random_flag = _i.hsr_filing_test_type.name == "SoP_RND"
    if not _hsr_test_random_flag:
        hft_seed = None
    elif not _v:
        hft_seed = SeedSequence(pool_size=8)
    else:
        idx = 2
        if _random_price_variation_flag:
            idx += 1
        if not _uniform_share_distribution_flag:
            idx += 1
        hft_seed = _v[idx]

    return SeedSequenceData(
        share=shr_seed,
        pcm=pcm_seed,
        fcounts=fct_seed,
        price=pri_seed,
        hsr_filing_test=hft_seed,
    )


@YAML.register_class
@define(kw_only=True, slots=True, frozen=True)
class MarketSample:
    """Parameter specification for market data generation."""

    share_spec: ShareSpec = field(
        default=ShareSpec(SHRDistribution.UNI),
        validator=validators.instance_of(ShareSpec),
    )
    """Market-share specification, see :class:`.ShareSpec`"""

    pcm_spec: PCMSpec = field(
        default=PCMSpec(PCMDistribution.UNI), validator=validators.instance_of(PCMSpec)
    )
    """Margin specification, see :class:`.PCMSpec`"""

    @pcm_spec.validator
    def _psv(self, _: Attribute[PCMSpec], _v: PCMSpec, /) -> None:
        if (
            self.share_spec.recapture_form == RECForm.FIXED
            and _v.pcm_restriction == PCMRestriction.MNL
        ):
            raise ValueError(
                f'Specification of "PCMSpec.pcm_restriction", as {PCMRestriction.MNL!r} '
                f'requires that "ShareSpec.recapture_form" be {RECForm.INOUT!r} '
                f"or {RECForm.OUTIN!r}, not {RECForm.FIXED!r} as presently specified"
            )

    price_spec: PriceSpec = field(validator=validators.instance_of(PriceSpec))
    """Price specification, see :class:`.PriceSpec`"""

    @price_spec.default
    def _price_spec_default(self) -> PriceSpec:
        return (
            PriceSpec.COST_SYM
            if self.pcm_spec.pcm_restriction == PCMRestriction.MNL
            else PriceSpec.PRICE_SYM
        )

    hsr_filing_test_type: HSRFilingTest = field(
        default=HSRFilingTest.NONE, validator=validators.instance_of(HSRFilingTest)
    )
    """Method for modeling HSR filing thresholds, see :class:`.HSRFilingTest`"""

    @hsr_filing_test_type.validator
    def _hsftv(self, _: Attribute[HSRFilingTest], _v: HSRFilingTest, /) -> None:
        if (
            self.hsr_filing_test_type == HSRFilingTest.SoP_NTH
        ) and self.share_spec.distribution == SHRDistribution.UNI:
            raise ValueError(
                f'Specification of "HSRFilingTest", as {HSRFilingTest.SoP_NTH!r} '
                f"with market shares having uniform distribution is infeasible. "
                f"With uniformly distributed markets shares, share and prices are "
                f"defined for the merging firms only, so no n-th firm revenues can "
                f"be computed for implementing this form of HSR filing test."
            )

    hmt_flag: HMTSpec | Literal[False] = field(default=False)
    """Whether draws are to be tested for a relevant antitrust market.

    If a HMTSpec object is provided, only those draws are returned which contain
    a relevant antitrust market exists for the given product (the first product,
    by default).
    See :class:`.HMTSpec`
    """

    @hmt_flag.validator
    def _hmtv(
        self, _: Attribute[HMTSpec | Literal[False]], _v: HMTSpec | Literal[False], /
    ) -> None:
        if _v and self.share_spec.distribution == SHRDistribution.UNI:
            raise ValueError(
                "It is not possible to implement the hypothetical market test without "
                "information on the non-merging firms, which is the case when shares "
                "are specified as having uniform distribution."
                "Respecify with shares having Dirichlet distribution, SHRDistribution.DIR_FLAT."
            )
        if _v and _v.product_index not in {0, 1}:
            print(
                "WARNING: "
                "For calculating enforcement counts, the first two firms in the generated data "
                "are taken to be the merging firms. Thus specifying attribute, "
                f'"HMTSpec.product_index" as {_v.product_index} is atypical. '
                "Now you know."
            )

    sample_size: int = field(default=10**6, validator=validators.instance_of(int))
    """number of draws to simulate"""

    seed_data: SeedSequenceData = field(
        converter=Converter(_seed_data_conv, takes_self=True)  # type: ignore
    )
    """sequence of SeedSequences to ensure replicable data generation with
    appropriately independent random streams
    """

    @seed_data.default
    def _dsd(self) -> SeedSequenceData | None:
        return _seed_data_conv(None, self)

    @seed_data.validator
    def _sdv(
        _i: MarketSample, _: Attribute[SeedSequenceData], _v: SeedSequenceData, /
    ) -> None:
        if _i.share_spec.distribution == SHRDistribution.UNI and _v.fcounts:
            raise ValueError(
                "Attribute, seed_data.fcounts is ignored as irrelevant when "
                "market shares are drawn with Uniform distribution. "
                "Set seed_data.fcounts to None and retry."
            )

        if not _i.price_spec.name.endswith("RND") and _v.price is not None:
            raise ValueError(
                "Attribute, seed_data.price is ignored as irrelevant unless "
                "prices are asymmetric and uncorrelated and price-cost margins "
                "are also not symmetric. Set seed_data.price to None and retry."
            )
        elif _i.price_spec.name.endswith("RND") and _v.price is None:
            raise ValueError(
                "Attribute, seed_data.price is required when prices are generated "
                "from random draws. Set seed_data.price and retry."
            )

        if _i.hsr_filing_test_type.name == "SoP_RND" and _v.hsr_filing_test is None:
            raise ValueError(
                "Attribute, seed_data.hsr_filing_test is required when a HSR filing "
                "test vector is drawn from a Beta distribution. Set "
                "seed_data.hsr_filing_test and retry."
            )

    nthreads: int = field(default=NTHREADS, validator=validators.instance_of(int))
    """number of parallel threads to use"""

    dataset: MarketsData | None = field(default=None, init=False)

    enforcement_counts: UPPTestsCounts | None = field(default=None, init=False)

    def generate_sample(self) -> None:
        """Populate :attr:`dataset` with generated data.

        Returns
        -------
        None

        """
        _nthreads = self.nthreads
        _shr_scale, _iter_count, _trunc_size, _rng_seed_data = _ll_setup(self)

        if _rng_seed_data is None:
            object.__setattr__(
                self,
                "dataset",
                _markets_sampler(
                    self,
                    sample_size=(_shr_scale, self.sample_size),
                    seed_data=self.seed_data,
                    nthreads=_nthreads,
                ),
            )
            return

        with parallel_config(
            backend="threading",
            n_jobs=min(_nthreads, _iter_count),
            return_as="generator",
        ):
            _res_list = Parallel()(
                delayed(_markets_sampler)(
                    self,
                    sample_size=(
                        _shr_scale,
                        _trunc_size
                        if (1 + _rng_seed_data_ch.share.spawn_key[-1]) == _iter_count
                        else SUBSAMPLE_SIZE,
                    ),
                    seed_data=_rng_seed_data_ch,
                    nthreads=_nthreads,
                )
                for _rng_seed_data_ch in _rng_seed_data
            )

        object.__setattr__(
            self,
            "dataset",
            MarketsData(**{
                _k.name: np.vstack([getattr(_j, _k.name) for _j in _res_list])
                for _k in MarketsData.__attrs_attrs__
            }),
        )

    def test_enforcement(
        self, _enf_parm_vec: MGThresholds, _upp_test_regime: UPPTestRegime, /
    ) -> None:
        """Populate :attr:`enforcement_counts` with estimated UPP test counts.

        If the :attr:`dataset` attribute is not None, the counts are
        computed on the :attr:`dataset` attribute. Otherwise, enforcement counts
        are computed in parallel on data generated on the fly. Note here that if
        the sample size is not an even multiple of SUBSAMPLE_SIZE,
        the sample on which enforcement counts are computed will be larger than
        the specified sample size, by up to SUBSAMPLE_SIZE draws less 1.

        Parameters
        ----------
        _enf_parm_vec
            Threshold values for various Guidelines criteria

        _upp_test_regime
            Specifies whether to analyze enforcement, clearance, or both
            and the GUPPI and diversion ratio aggregators employed, with
            default being to analyze enforcement based on the maximum
            merging-firm GUPPI and maximum diversion ratio between the
            merging firms

        Returns
        -------
        None

        """
        object.__setattr__(
            self,
            "enforcement_counts",
            _sim_enf_cnts_ll(self, _enf_parm_vec, _upp_test_regime)
            if self.dataset is None
            else compute_upp_test_counts(
                self.share_spec, self.dataset, _enf_parm_vec, _upp_test_regime
            ),
        )

    def to_archive(
        self,
        zip_: zipfile.ZipFile,
        _zipsubdir: str = "",
        /,
        *,
        save_dataset: bool = False,
    ) -> None:
        """Serialize market sample to Zip archive."""
        zpath = zipfile.Path(zip_, at=_zipsubdir)  # type: ignore[arg-type,unused-ignore]
        name_root = f"{PKG_NAME}_market_sample"

        with (zpath / f"{name_root}.yaml").open("w") as _yfh:
            YAML.dump(self, _yfh)

        if save_dataset:
            if self.dataset is None and self.enforcement_counts is None:
                raise ValueError(
                    "No dataset and/or enforcement counts available for saving. "
                    "Generate some data or set save_dataset to False to proceed."
                )

            else:
                if self.dataset is not None:
                    with (zpath / f"{name_root}_dataset.h5").open("wb") as _hfh:
                        _hfh.write(self.dataset.to_h5bin())

                if self.enforcement_counts is not None:
                    with (zpath / f"{name_root}_enforcement_counts.yaml").open(
                        "w"
                    ) as _yfh:
                        YAML.dump(self.enforcement_counts, _yfh)

        if not zip_.namelist():
            raise AssertionError("Serialization failed. Archive is empty.")

        elif not zpath.joinpath(f"{name_root}.yaml").is_file():
            raise AssertionError(
                "Serialization failed. Archive has no serialized MarketSample class instance."
            )

        if (
            save_dataset
            and not zpath.joinpath(f"{name_root}_enforcement_counts.yaml").is_file()
        ):
            raise AssertionError(
                "Serialization failed. Archive has no serialized enforcement counts data."
            )

    @staticmethod
    def from_archive(
        zip_: zipfile.ZipFile, _subdir: str = "", /, *, restore_dataset: bool = False
    ) -> MarketSample:
        """Deserialize market sample from Zip archive."""
        zpath = zipfile.Path(zip_, at=_subdir)  # type: ignore[arg-type,unused-ignore]
        name_root = f"{PKG_NAME}_market_sample"

        if not zip_.namelist():
            raise ValueError("Archive is empty.")

        elif not (zpath / f"{name_root}.yaml").is_file():
            raise ValueError("Archive has no serialized MarketSample class instance.")

        market_sample_: MarketSample = YAML.load(
            (zpath / f"{name_root}.yaml").read_text()
        )

        if restore_dataset:
            _dt = (_dp := zpath / f"{name_root}_dataset.h5").is_file()
            _et = (_ep := zpath / f"{name_root}_enforcement_counts.yaml").is_file()
            if not (_dt or _et):
                print(f"Archive: {zpath}".rstrip("/"))
                raise ValueError(
                    "Archive has no sample data to restore. "
                    "Delete second argument, or set it False, and rerun."
                )
            else:
                if _dt:
                    with _dp.open("rb") as _hfh:
                        object.__setattr__(
                            market_sample_, "dataset", MarketsData.from_h5f(_hfh)
                        )
                if _et:
                    object.__setattr__(
                        market_sample_, "enforcement_counts", YAML.load(_ep.read_text())
                    )
        return market_sample_

    @classmethod
    def to_yaml(
        cls, _r: yaml.representer.RoundTripRepresenter, _d: MarketSample
    ) -> yaml.MappingNode:
        """Serialize market sample to YAML representation."""
        retval: yaml.MappingNode = _r.represent_mapping(
            f"!{cls.__name__}",
            {
                _a.name: getattr(_d, _a.name)
                for _a in _d.__attrs_attrs__
                if _a.name not in {"dataset", "enforcement_counts"}
            },
        )
        return retval

    @classmethod
    def from_yaml(
        cls, _c: yaml.constructor.RoundTripConstructor, _n: yaml.MappingNode
    ) -> MarketSample:
        """Deserialize market sample from YAML representation."""
        return cls(**yaml_rt_mapper(_c, _n))


def _ll_setup(
    _market_sample: MarketSample, /
) -> tuple[float, int, int | None, tuple[SeedSequenceData] | None]:
    """Set up parallelized data generation."""
    _sample_size = _market_sample.sample_size
    _seed_data = _market_sample.seed_data

    # Scale up sample size to offset discards based on specified criteria
    # 1. HSR filing test type
    shr_sample_size = (
        _sample_size
        * {"SoP_TEN": 3, "SoP_NTH": 1.9, "SoP_RND": 1.5}.get(
            _market_sample.hsr_filing_test_type.name, 1
        )
        * (2.5 if _market_sample.share_spec.distribution.name == "UNI" else 1)
        * (
            {"SYM": 2.5, "MNL": 2.0}.get(
                _market_sample.pcm_spec.pcm_restriction.name, 1
            )
            if (
                _market_sample.hsr_filing_test_type.name == "SoP_RND"
                and _market_sample.price_spec.name.endswith(("POS", "NEG"))
            )
            else 1
        )
        * (
            1.2
            if (
                _market_sample.pcm_spec.pcm_restriction == PCMRestriction.SYM
                and _market_sample.hsr_filing_test_type == HSRFilingTest.SoP_NTH
                and _market_sample.price_spec
                in {PriceSpec.COST_NEG, PriceSpec.COST_POS, PriceSpec.PRICE_NEG}
            )
            else 1
        )
    )

    # 2. PCM restriction for equilibrium conditions with MNL demand
    shr_sample_size *= (
        (
            1.0
            if _market_sample.price_spec.name.startswith("COST_SYM")
            else (
                3.0
                if _market_sample.share_spec.distribution == SHRDistribution.UNI
                else 2.0
            )
            * (15 if _market_sample.price_spec.name.endswith(("POS", "NEG")) else 1)
        )
        if _market_sample.pcm_spec.pcm_restriction == PCMRestriction.MNL
        else 1
    )

    # Scale up further for HMT restriction
    # Estimated smallest retention rate without MNL restriction is 0.85
    if _market_sample.hmt_flag:
        match _market_sample.hsr_filing_test_type:
            case HSRFilingTest.SoP_RND:
                shr_sample_size *= 1 / 0.25
            case HSRFilingTest.SoP_TEN:
                shr_sample_size *= 1 / 0.3
            case HSRFilingTest.SoP_NTH:
                shr_sample_size *= 1 / 0.133
            case _:
                shr_sample_size *= 1 / 0.7

    _shr_scale = shr_sample_size / _sample_size

    _iter_count = ceil(_sample_size / SUBSAMPLE_SIZE)
    if _iter_count == 1:
        return _shr_scale, _iter_count, None, None
    else:
        _trunc_size = (
            _market_sample.sample_size % SUBSAMPLE_SIZE
            if (_iter_count * SUBSAMPLE_SIZE) > _market_sample.sample_size
            else SUBSAMPLE_SIZE
        )

        # The expression below does the following:
        #  1. Spawn iter_count number of seed sequences for each seed
        #     (attribute) in seed_data
        #  2. Zip the i-th seeds across the spawns into a generator over
        #     the spawns
        #  3. Map the SeedSequenceData constructor over each sequence of i-th
        #     spawns to get iter_count seed_sequence_data used to seed the iterations
        _rng_seed_data = tuple(
            starmap(
                SeedSequenceData,
                zip(
                    *[
                        _s.spawn(_iter_count) if _s else [None] * _iter_count
                        for _s in (
                            getattr(_seed_data, _a.name)
                            for _a in _seed_data.__attrs_attrs__
                        )
                    ],
                    strict=True,
                ),
            )
        )

        return _shr_scale, _iter_count, _trunc_size, _rng_seed_data  # type: ignore[return-value]


def _markets_sampler(
    _market_sample: MarketSample,
    /,
    *,
    sample_size: int | tuple[float, int],
    seed_data: SeedSequenceData,
    nthreads: int,
) -> MarketsData:
    """
    Generate share, diversion ratio, price, and margin data for MarketSpec.

    This function is called within a parallel context. The keyword arguments are specific
    to the subset of the sample in each of the parallel operations.

    sample_size:
        Number of draws to generate. If a tuple, the first element is the
        multiplier determining the initial sample size, while the second number is
        the final sample size.

    seed_data:
        Seed data to ensure independent and replicable draws.

    nthreads:
        Number of parallel threads to use for random number generation.


    Returns
    -------
    Merging firms' shares, margins, etc. for each hypothetical  merger
    in the sample

    """
    _shr_scale, _final_ssz = (
        (1.0, sample_size) if isinstance(sample_size, int) else sample_size
    )
    _shr_ssz = int(_shr_scale * _final_ssz)

    # Generate share data
    mktshr_array, aggr_choice_prob = market_share_sampler(
        _market_sample.share_spec, _shr_ssz, seed_data, nthreads
    )
    # Generate merging-firm price and PCM data
    price_array, pcm_array, mnl_test, hsr_filing_test = prices_sampler(
        _market_sample.share_spec,
        _market_sample.pcm_spec,
        _market_sample.price_spec,
        _market_sample.hsr_filing_test_type,
        mktshr_array,
        aggr_choice_prob,
        seed_data,
        nthreads,
    )

    if _shr_scale > 1.0:
        mnl_test_rows = mnl_test * hsr_filing_test

        mktshr_array = mktshr_array[mnl_test_rows]
        pcm_array = pcm_array[mnl_test_rows]
        price_array = price_array[mnl_test_rows]
        aggr_choice_prob = (
            aggr_choice_prob
            if _market_sample.share_spec.recapture_form == RECForm.FIXED
            else aggr_choice_prob[mnl_test_rows]
        )

        del mnl_test_rows

    if (
        _market_sample.share_spec.distribution == SHRDistribution.UNI
        or not _market_sample.hmt_flag
    ):
        return MarketsData(
            shares=mktshr_array[:_final_ssz],
            margins=pcm_array[:_final_ssz],
            prices=price_array[:_final_ssz],
            aggregate_choice_probability=aggr_choice_prob[:_final_ssz],
        )

    # Pre-test:
    # _diversion_ratios = compute_merging_firm_diversion_ratios(
    #     _market_sample.share_spec.recapture_form,
    #     _market_sample.share_spec.recapture_rate,
    #     mktshr_array,
    #     aggr_choice_prob,
    # )

    _hmt_mkt_flag, _hmt_prod_flags = _recapture_hmt(
        _market_sample.hmt_flag, mktshr_array, aggr_choice_prob, price_array, pcm_array
    )

    if _market_sample.hmt_flag.recompute_shares:
        mktshr_array_hmt, aggr_choice_prob_hmt = (
            np.divide(
                _mr := mktshr_array * _hmt_prod_flags,
                _ma := np.einsum("ij->i", _mr)[:, None],
            ),
            aggr_choice_prob * _ma,
        )

        # Pre-test:
        # _diversion_ratios_hmt = compute_merging_firm_diversion_ratios(
        #     _market_sample.share_spec.recapture_form,
        #     _market_sample.share_spec.recapture_rate,
        #     mktshr_array_hmt,
        #     aggr_choice_prob_hmt,
        # )

        # Test:
        # if not np.allclose(_diversion_ratios, _diversion_ratios_hmt):
        #     raise ValueError("Market definition analysis is incorrect.")
    else:
        mktshr_array_hmt, aggr_choice_prob_hmt = mktshr_array, aggr_choice_prob
    del mktshr_array

    return MarketsData(
        shares=mktshr_array_hmt[_hmt_mkt_flag][:_final_ssz],
        margins=pcm_array[_hmt_mkt_flag][:_final_ssz],
        prices=price_array[_hmt_mkt_flag][:_final_ssz],
        aggregate_choice_probability=aggr_choice_prob_hmt[_hmt_mkt_flag][:_final_ssz],
    )


def _recapture_hmt(
    _hmt_flag: HMTSpec,
    _market_shares: ArrayDouble,
    _aggr_purch_prob: ArrayDouble,
    _prices: ArrayDouble,
    _margins: ArrayDouble,
    /,
) -> tuple[ArrayBoolean, ArrayBoolean]:
    """
    Test market definition using opportunity costs.

    Parameters
    ----------
    _hmt_flag
        Specification for restricting draws to those with a revelant antitrust
        market for the given product, using the specified SSNIP level and
        passthrough rate and whether to recompute shares.

    _market_shares
        All-firm market shares.

    _aggr_purch_prob
        1 minus probability that the outside good is chosen; converts
        market shares to choice probabilities by multiplication.

    _prices
        All-firm prices.

    _margins
        All-firm marginal costs.

    Returns
    -------
        Merging-firm diversion ratios for mergers in the sample.

    Raises
    ------
    ValueError
        If the firm with the smaller share does not have the larger
        diversion ratio between the merging firms.

    """
    _choice_probabilities = np.einsum("ij,ij->ij", _aggr_purch_prob, _market_shares)

    _prod_idx = _hmt_flag.product_index

    divratio_1j = np.divide(
        _choice_probabilities, 1 - _choice_probabilities[:, [_prod_idx]]
    )
    divratio_1j[:, [_prod_idx]] = np.zeros_like(divratio_1j[:, [_prod_idx]])

    guppi_1 = np.divide(
        np.einsum("ij,ij,ij->ij", divratio_1j, _margins, _prices),
        _prices[:, [_prod_idx]],
    )

    return (
        _hmt_product_flag_from_opportunity_cost(
            guppi_1,
            divratio_1j,
            passthru_rate=_hmt_flag.passthrough_rate,
            SSNIP=_hmt_flag.ssnip_level,
        )
        if _hmt_flag.recompute_shares
        else (
            ArrayBoolean(
                np.einsum("ij->i", guppi_1)
                > (_hmt_flag.ssnip_level / _hmt_flag.passthrough_rate)
            ),
            EMPTY_ARRAYBOOLEAN,
        )
    )


def _hmt_product_flag_from_opportunity_cost(
    _guppi: ArrayDouble, _divratio: ArrayDouble, /, passthru_rate: float, SSNIP: float
) -> tuple[ArrayBoolean, ArrayBoolean]:
    """Test market definition and mask products in each market.

    Reference:

    See, Joseph Farrell and Carl Shapiro. 2010. "Recapture, Pass-Through and
    Market Definition." Antitrust Law Journal | January 2010 , Vol 76(3): pp. 585-604
    """
    # Step1: Find the ordering of draws by diversion ratio from "first" merging firm to  by descending order of diversion-ratio
    #  second merging firm, and to non-merging products in descending order
    _divratio_outer = _divratio[:, 2:]
    _divratio_outer_shape = _divratio_outer.shape

    _divratio_outer_order = np.hstack((_divratio_outer.argsort(axis=1)[:, ::-1],))

    # Step2: rearrange GUPPI array (opportunity cost array) by the
    # "diversion-ratio ordering" defined above, and test find the row index,
    # if any, at which the cumulative sum of opportunity costs first meets or
    # exceeds the critical opportunity cost
    _guppi_sorted_for_hmt = np.hstack((
        _guppi[:, :2],
        np.take_along_axis(_guppi[:, 2:], _divratio_outer_order, axis=1),
    ))

    _guppi_for_hmt = _guppi_sorted_for_hmt.cumsum(axis=1) > (SSNIP / passthru_rate)
    del _guppi_sorted_for_hmt

    # Step 2a: First, create a filter based on whether the cumulative opportunity cost
    # through the n-th firm in the market exceeds the critical opportunity cost:
    # if the hmt fails for the putative market excluding the outside good,
    # the sampled "market" includes no relevant antitrust market

    _hmt_market_flag = _guppi_for_hmt[:, -1]

    # Step 2b: Now crate a product selector, across all draws, with
    # a 1 for products in a relevant market and 0 for those not in
    # a relevant market, as defined by the HMT
    hmt_outer_product_flag = np.indices(_divratio_outer_shape)[-1] <= (
        -2 + _guppi_for_hmt.argmax(axis=1, keepdims=True)
    )
    del _guppi_for_hmt

    np.put_along_axis(
        hmt_outer_product_flag,
        _divratio_outer_order,
        axis=1,
        values=hmt_outer_product_flag,
    )

    return ArrayBoolean(_hmt_market_flag), ArrayBoolean(
        np.hstack((np.ones_like(_guppi[:, :2]), hmt_outer_product_flag))
    )


def _sim_enf_cnts_ll(
    _market_sample: MarketSample,
    _enf_parm_vec: MGThresholds,
    _enf_test_regime: UPPTestRegime,
    /,
) -> UPPTestsCounts:
    """Parallelize data-generation and testing.

    Parameters
    ----------
    _market_sample
        Market sample to test against
    _enf_parm_vec
        Guidelines thresholds to test against

    _sim_test_regime
        Configuration to use for testing

    Returns
    -------
        Arrays of enforcement counts or clearance counts by firm count,
        ΔHHI and concentration zone

    """
    if (
        _market_sample.share_spec.recapture_form != RECForm.OUTIN
        and _market_sample.share_spec.recapture_rate != _enf_parm_vec.rec
    ):
        raise ValueError(
            "{} {} {}".format(
                f"Recapture rate from market sample spec, {_market_sample.share_spec.recapture_rate}",
                f"must match the value, {_enf_parm_vec.rec}",
                "the guidelines thresholds vector.",
            )
        )

    _nthreads = _market_sample.nthreads
    _shr_scale, _iter_count, _trunc_size, _rng_seed_data = _ll_setup(_market_sample)

    if _rng_seed_data is None:
        return _sim_enf_cnts(
            _market_sample,
            _enf_parm_vec,
            _enf_test_regime,
            sample_size=(_shr_scale, _market_sample.sample_size),
            seed_data=_market_sample.seed_data,
            nthreads=_nthreads,
        )

    with parallel_config(
        backend="threading", n_jobs=min(_nthreads, _iter_count), return_as="generator"
    ):
        _res_list = Parallel()(
            delayed(_sim_enf_cnts)(
                _market_sample,
                _enf_parm_vec,
                _enf_test_regime,
                sample_size=(
                    _shr_scale,
                    _trunc_size
                    if (1 + _rng_seed_data_ch.share.spawn_key[-1]) == _iter_count
                    else SUBSAMPLE_SIZE,
                ),
                seed_data=_rng_seed_data_ch,
                nthreads=_nthreads,
            )
            for _rng_seed_data_ch in _rng_seed_data
        )

    _res_list_stacks = [
        ArrayBIGINT(np.vstack([getattr(_j, _k) for _j in _res_list]))
        for _k in (f"{StatsGroup.FC}", f"{StatsGroup.DL}", f"{StatsGroup.HD}")
    ]

    return UPPTestsCounts(*[
        (EMPTY_ARRAYBIGINT if not _g.any() else compute_enforcement_counts(_g, _h))
        for _g, _h in zip(
            _res_list_stacks, (StatsGroup.FC, StatsGroup.DL, StatsGroup.HD), strict=True
        )
    ])


def _sim_enf_cnts(
    _market_sample: MarketSample,
    _upp_test_parms: MGThresholds,
    _upp_test_regime: UPPTestRegime,
    /,
    *,
    sample_size: tuple[float, int],
    seed_data: SeedSequenceData,
    nthreads: int = NTHREADS,
) -> UPPTestsCounts:
    """Generate market data and compute UPP test counts on same.

    Parameters
    ----------
    _upp_test_parms
        Guidelines thresholds for testing UPP and related statistics

    _upp_test_regime
        Configuration to use for testing; UPPTestsRegime object
        specifying whether investigation results in enforcement, clearance,
        or both; and aggregation methods used for GUPPI and diversion ratio
        measures

    sample_size
        Number of draws to generate

    seed_data
        List of seed sequences, to assure independent samples in each thread

    nthreads
        Number of parallel processes to use

    Returns
    -------
        UPPTestCounts object with  of test counts by firm count, ΔHHI and concentration zone

    """
    market_data_sample = _markets_sampler(
        _market_sample, sample_size=sample_size, seed_data=seed_data, nthreads=nthreads
    )

    return compute_upp_test_counts(
        _market_sample.share_spec, market_data_sample, _upp_test_parms, _upp_test_regime
    )
