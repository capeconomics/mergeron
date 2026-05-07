"""Non-public functions called in data_generation.py."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from joblib import Parallel
from joblib import delayed

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.random import SeedSequence
    from numpy.typing import NDArray

    from ..core import EmpiricalMarginData

from .. import EMPTY_ARRAYDOUBLE
from .. import VERSION
from .. import ArrayBIGINT
from .. import ArrayBoolean
from .. import ArrayDouble
from .. import ArrayFloat
from .. import ArrayUINT8
from .. import RECForm
from ..core.pseudorandom_numbers import MultithreadedRNG
from ..core.pseudorandom_numbers import prng
from . import DEFAULT_DIR_COND_DIST_PARMS
from . import HSR_BETA_PARMS
from . import HSR_RATIO
from . import SUBSAMPLE_SIZE
from . import HSRFilingTest
from . import PCMDistribution
from . import PCMRestriction
from . import PCMSpec
from . import PriceSpec
from . import SeedSequenceData
from . import ShareSpec
from . import SHRDistribution

__version__ = VERSION


def market_share_sampler(
    _share_spec: ShareSpec,
    _sample_size: int,
    _seed_data: SeedSequenceData,
    _nthreads: int,
    /,
) -> tuple[ArrayDouble, ArrayDouble]:
    """Generate share data.

    Parameters
    ----------
    _share_spec
        Class specifying parameters for generating market share data

    _sample_size
        Number of market shares to generate

    _seed_data
        List of seed sequences, including seeds for generating firm-counts
        and market shares from firm-counts

    _nthreads
        Must be specified for generating repeatable random streams

    Returns
    -------
    Generated shares and aggregate purchase probabilities.

    """
    dist_type_mktshr, dist_parms_mktshr, share_lower_bound, recapture_form = (
        getattr(_share_spec, _f)
        for _f in ("distribution", "parameters", "lower_bound", "recapture_form")
    )

    _fcount_rng_seed_seq = _seed_data.fcounts
    _mktshr_rng_seed_seq = _seed_data.share

    aggregate_choice_probability = EMPTY_ARRAYDOUBLE
    if dist_type_mktshr == SHRDistribution.UNI:
        _market_shares = market_share_sampler_uniform(
            dist_parms_mktshr,
            _sample_size,
            share_lower_bound,
            _mktshr_rng_seed_seq,
            _nthreads,
        )
    elif dist_type_mktshr.name.startswith("DIR_"):
        _market_shares, aggregate_choice_probability = (
            _market_share_sampler_dirichlet_pooled(
                _share_spec,
                _sample_size,
                _fcount_rng_seed_seq,
                _mktshr_rng_seed_seq,
                _nthreads,
            )
        )

    else:
        raise ValueError(
            f'Unexpected type, "{dist_type_mktshr}" for share distribution.'
        )

    # If recapture_form == "inside-out", recalculate _aggr_purch_prob
    if recapture_form == RECForm.INOUT:
        if _share_spec.recapture_rate is None:
            # type-narrowing; enforced at _share_spec instantiation
            raise ValueError(
                "Recapture rate must be specified for inside-out recapture."
            )
        # treating r_bar as the recapture rate for the smaller merging firm, 1 - π_0,
        aggregate_choice_probability = (_r := _share_spec.recapture_rate) / (
            1 - (1 - _r) * _market_shares[:, :2].min(axis=1, keepdims=True)
        )

    return _market_shares, aggregate_choice_probability


def market_share_sampler_uniform(
    _dist_parms_mktshr: ArrayFloat,
    _ssz: int,
    _share_lower_bound: float,
    _mktshr_rng_seed_seq: SeedSequence,
    _nthreads: int,
    /,
) -> ArrayDouble:
    """Generate merging-firm shares from Uniform distribution on the 3-D simplex.

    Parameters
    ----------
    _ssz
        size of sample to be drawn

    _mktshr_rng_seed_seq
        seed for rng, so results can be made replicable

    _nthreads
        number of threads for random number generation

    Returns
    -------
        generated market shares

    """
    mktshr_array = ArrayDouble(np.empty((_ssz, 2)))

    MultithreadedRNG(
        mktshr_array,
        distribution="Uniform",
        parameters=_dist_parms_mktshr,
        seed_sequence=_mktshr_rng_seed_seq,
        nthreads=_nthreads,
    ).fill()

    # Convert draws on U[0, 1] to Uniformly-distributed draws on simplex, s_1 + s_2 <= 1
    mktshr_array = np.hstack((
        (_u := np.sort(mktshr_array, axis=1))[:, [0]],
        _u[:, [1]] - _u[:, [0]],
    ))

    # Apply lower bound, if given
    if _share_lower_bound:
        return ArrayDouble(
            _share_lower_bound + (1 - _share_lower_bound * 3) * mktshr_array
        )
    else:
        return ArrayDouble(mktshr_array[mktshr_array.min(axis=1) > 0])


def _market_share_sampler_dirichlet_pooled(
    _share_spec: ShareSpec,
    _ssz: int,
    _fcount_rng_seed_seq: SeedSequence | None,
    _mktshr_rng_seed_seq: SeedSequence,
    _nthreads: int,
    /,
) -> tuple[ArrayDouble, ArrayDouble]:
    """Dirichlet-distributed shares with multiple firm-counts.

    Firm-counts may be specified as having Uniform distribution over the range
    of firm counts, or a set of probability weights may be specified. In the
    latter case the proportion of draws for each firm-count matches the
    specified probability weight.

    Parameters
    ----------
    _share_spec
        market share specification

    _ssz
        sample size to be drawn

    _fcount_rng_seed_seq
        seed firm count rng, for replicable results

    _mktshr_rng_seed_seq
        seed market share rng, for replicable results

    _nthreads
        number of threads for parallelized random number generation

    Returns
    -------
        array of market shares and other market statistics

    """
    (
        _share_distribution,
        _firm_count_wts,
        _share_parameters,
        _share_lower_bound,
        _recapture_form,
    ) = (
        getattr(_share_spec, _f)
        for _f in (
            "distribution",
            "firm_counts_weights",
            "parameters",
            "lower_bound",
            "recapture_form",
        )
    )

    min_choice_wt = (
        0.03 if _share_distribution == SHRDistribution.DIR_FLAT_CONSTR else 0.00
    )
    choice_wts = np.divide(
        _nr := np.array([_w if _w > min_choice_wt else 0.0 for _w in _firm_count_wts]),
        _nr.sum(),
    )
    fcount_keys = ArrayUINT8(2 + np.arange(len(_firm_count_wts), dtype=np.uint8))

    fc_max = fcount_keys[-1]
    mktshr_array = ArrayDouble(np.empty((_ssz, fc_max)))
    aggr_choice_prob = EMPTY_ARRAYDOUBLE

    mktshr_seed_seq_ch = _mktshr_rng_seed_seq.spawn(len(fcount_keys))

    if _recapture_form == RECForm.OUTIN:
        fcount_keys += 1
        aggr_choice_prob = ArrayDouble(np.empty((_ssz, 1)))
        if (_s := len(_share_parameters)) < (_l := fcount_keys[-1]):
            _share_parameters = np.concatenate([
                _share_parameters,
                _share_parameters[-1:] * (_l - _s),
            ])

    # fcounts = np.repeat(
    #     fcount_keys,
    #     _w := ArrayBIGINT(np.ceil(100 * choice_wts / choice_wts[choice_wts != 0].min())),
    # )[np.random.default_rng(_fcount_rng_seed_seq).integers(_w.sum(), size=(_ssz, 1))]
    fcounts = (
        _w := (
            fcount_keys
            if all(
                len(_v) == 1 and _w == len(choice_wts)
                for _v, _w in [np.unique(choice_wts, return_counts=True)]
            )
            else np.repeat(
                fcount_keys,
                ArrayBIGINT(
                    np.ceil(100 * choice_wts / choice_wts[choice_wts != 0].min())
                ),
            )
        )
    )[np.random.default_rng(_fcount_rng_seed_seq).integers(len(_w), size=_ssz)]

    for _f_val, _f_seed in zip(fcount_keys, mktshr_seed_seq_ch, strict=True):
        fcounts_match_rows = np.where(fcounts == _f_val)[0]
        _ssz_sub = len(fcounts_match_rows)
        if _ssz_sub == 0:
            continue

        dir_alphas_test = _dir_alphas_builder(
            _share_parameters, _f_val, _share_distribution
        )

        mktshr_array_f = market_share_sampler_dirichlet(
            dir_alphas_test, _share_lower_bound, _ssz_sub, _f_seed, _nthreads
        )

        if _recapture_form == RECForm.OUTIN:
            aggr_choice_prob_f = 1 - mktshr_array_f[:, [-1]]

            mktshr_array_f = mktshr_array_f[:, :-1] / aggr_choice_prob_f
            # If recapture_form is not 'outside_in', then
            # aggr_choice_prob is calculated downstream, leave it empty
            aggr_choice_prob[fcounts_match_rows] = aggr_choice_prob_f

        # Push data for present sample to parent array
        mktshr_array[fcounts_match_rows] = np.pad(
            mktshr_array_f, ((0, 0), (0, fc_max - mktshr_array_f.shape[1])), "constant"
        )

    return (mktshr_array, aggr_choice_prob)


def _dir_alphas_builder(
    _dir_alphas: NDArray[np.floating], _fcv: int, _dist_type: SHRDistribution
) -> ArrayFloat:

    if _dist_type == SHRDistribution.DIR_COND:
        _theta_1, _theta_2, _Theta = _dir_alphas
        _, _r = 2, _fcv - 2  # merging-firm count, non-merging firm-count
        return ArrayFloat(
            [_theta_1, _theta_2] + ([] if _fcv == 2 else [_Theta / _r] * _r)
        )

    return ArrayFloat(_dir_alphas[:_fcv] if _dir_alphas.size else [1.0] * _fcv)


def market_share_sampler_dirichlet(
    _dir_alphas: ArrayFloat,
    _share_lower_bound: float,
    _ssz: int,
    _mktshr_rng_seed_seq: SeedSequence,
    _nthreads: int,
    /,
) -> ArrayDouble:
    """Dirichlet-distributed shares with fixed firm-count.

    Parameters
    ----------
    _dir_alphas
        Shape parameters for Dirichlet distribution

    _ssz
        sample size to be drawn

    _mktshr_rng_seed_seq
        seed market share rng, for replicable results

    _nthreads
        number of threads for parallelized random number generation

    Returns
    -------
        array of market shares and other market statistics

    """
    _cond_dir_test = all((
        np.array_equal(_dir_alphas[:2], DEFAULT_DIR_COND_DIST_PARMS[:2]),
        np.isclose(_dir_alphas.sum(), DEFAULT_DIR_COND_DIST_PARMS.sum()),
    ))

    _ncols = len(_dir_alphas)
    mktshr_array = ArrayDouble(np.empty((_ssz, _ncols)))
    MultithreadedRNG(
        mktshr_array,
        distribution="Dirichlet",
        parameters=_dir_alphas,
        seed_sequence=_mktshr_rng_seed_seq,
        nthreads=_nthreads,
    ).fill()

    if _share_lower_bound and not _cond_dir_test:
        return ArrayDouble(
            _share_lower_bound + (1 - _share_lower_bound * _ncols) * mktshr_array
        )
    else:
        return ArrayDouble(mktshr_array[mktshr_array.min(axis=1) > 0.0])


def compute_merging_firm_diversion_ratios(
    _recapture_form: RECForm,
    _recapture_rate: float | None,
    _market_shares: ArrayDouble,
    _aggr_purch_prob: ArrayDouble,
    /,
) -> ArrayDouble:
    """
    Given merging-firm shares and related parameters, return diversion ratios.

    Parameters
    ----------
    _recapture_form
        Enum specifying Fixed (proportional), Inside-out, or Outside-in

    _recapture_rate
        If recapture is proportional or inside-out, the recapture rate
        for the firm with the smaller share.

    _market_shares
        Generated market shares.

    _aggr_purch_prob
        One (1) minus probability that the outside good is chosen; converts
        market shares to choice probabilities by multiplication.

    Returns
    -------
        Merging-firm diversion ratios for hypothetical mergers in the sample.

    Raises
    ------
    ValueError
        If the firm with the smaller share does not have the larger
        diversion ratio between the merging firms.

    """
    if _recapture_form == RECForm.FIXED:
        if _recapture_rate is None:
            raise ValueError(
                "If recapture form is, RECForm.FIXED, a recapture rate must be supplied."
            )
        return ArrayDouble(
            _recapture_rate * (_fsa := _market_shares[:, :2])[:, ::-1] / (1 - _fsa)
        )

    else:
        return ArrayDouble(
            np.divide(
                (
                    _ppa := np.einsum(
                        "ij,ij->ij", _aggr_purch_prob, _market_shares[:, :2]
                    )
                )[:, ::-1],
                1 - _ppa,
            )
        )


def compute_all_firm_diversion_ratios(
    _recapture_form: RECForm,
    _recapture_rate: float | None,
    _market_shares: ArrayDouble,
    _aggr_purch_prob: ArrayDouble,
    /,
) -> ArrayDouble:
    """
    Given shares and related parameters, return diversion ratios.

    Diversion ratios are returned in demand system layout, with 0's in
    the diagonals, and column-sums' giving market recapture rates:

    .. code-block:: python3

        rec = np.einsum(
            "ijk->ik", compute_all_firm_diversion_ratios(...)
        )


    Parameters
    ----------
    _recapture_form
        Enum specifying Fixed (proportional), Inside-out, or Outside-in

    _recapture_rate
        If recapture is proportional or inside-out, the recapture rate
        for the firm with the smaller share.

    _market_shares
        Generated market shares.

    _aggr_purch_prob
        The probability that one of the products in the sample ("market")
        is chosen; converts market shares to choice probabilities by multiplication.

    Returns
    -------
        All-firm diversion ratios for firms (products) in the sample.

    Raises
    ------
    ValueError
        If the firm with the smaller share does not have the larger
        diversion ratio between the merging firms.

    """
    if _recapture_form == RECForm.FIXED:
        if _recapture_rate is None:
            raise ValueError(
                "If recapture form is, RECForm.FIXED, a recapture rate must be supplied."
            )
        _purchase_prob = _market_shares
    else:
        if not _aggr_purch_prob.size:
            raise ValueError(
                "If recapture form is, RECForm.OUTIN or RECForm.INOUT, "
                "diversion ratio computation is infeasible unless "
                "aggregate choice probabilities are provided."
            )
        _purchase_prob = _aggr_purch_prob * _market_shares

    _n = _purchase_prob.shape[1]

    diversion_ratios = np.einsum(
        "ikj,ijk->ijk", _d := _purchase_prob[:, None, :], 1 / (1 - _d)
    )
    diversion_ratios[:, range(_n), range(_n)] = 0

    return ArrayDouble(
        _recapture_rate * diversion_ratios
        if _recapture_form == RECForm.FIXED
        else diversion_ratios
    )


def prices_sampler(
    _share_spec: ShareSpec,
    _pcm_spec: PCMSpec,
    _price_spec: PriceSpec,
    _hsr_filing_test_type: HSRFilingTest,
    _market_shares: ArrayDouble,
    _aggr_purch_prob: ArrayDouble,
    _seed_data: SeedSequenceData,
    _nthreads: int,
    /,
) -> tuple[ArrayDouble, ArrayDouble, ArrayBoolean, ArrayBoolean]:
    """Generate margin and price data for mergers in the sample.

    Parameters
    ----------
    _share_spec
        Enum specifying whether to use asymmetric or flat margins; see
        :class:`mergeron.gen.ShareSpec`.
    _pcm_spec
        Enum specifying whether to use asymmetric or flat margins. see
        :class:`mergeron.gen.PCMSpec`.

    _price_spec
        Enum specifying whether to use symmetric, positive, or negative
        margins; see :class:`mergeron.gen.PriceSpec`.

    _hsr_filing_test_type
        Enum specifying restriction, if any, to impose on market data sample
        to model HSR filing requirements; see :class:`mergeron.gen.HSRFilingTest`.

    _market_shares
        Generated market shares.

    _aggr_purch_prob
        1 minus probability that the outside good is chosen; converts
        market shares to choice probabilities by multiplication.

    _seed_data
        List of seed sequences, including seeds for generating prices and
        margins, and a HSR filing test vector, as needed.

    _nthreads
        Number of threads to use in generating price data.

    Returns
    -------
        Simulated price- and margin-data arrays, with MNL and HSR-filing test vectors, for
        mergers in the sample.
    """
    _pcm_rng_seed_seq, _price_rng_seed_seq, _hsr_rng_seed_seq = (
        getattr(_seed_data, _a) for _a in ("pcm", "price", "hsr_filing_test")
    )

    _lower_bound = _share_spec.lower_bound

    price_array = ArrayDouble(np.ones_like(_market_shares))
    pcm_array, mnl_test = (
        ArrayDouble(np.array([], float)),
        ArrayBoolean(np.array([], bool)),
    )

    nth_firm_share = EMPTY_ARRAYDOUBLE
    nth_firm_price = EMPTY_ARRAYDOUBLE

    share_uni_flag = _share_spec.distribution == SHRDistribution.UNI
    if not share_uni_flag:
        nth_firm_share = ArrayDouble(
            np.take_along_axis(
                _market_shares,
                np.einsum("ij->i", _market_shares > 0, dtype="<u1")[:, None] - 1,
                axis=1,
            )
        )

    if _price_spec == PriceSpec.PRICE_SYM and not share_uni_flag:
        nth_firm_price = ArrayDouble(np.ones_like(nth_firm_share))
    elif _price_spec in {PriceSpec.PRICE_POS, PriceSpec.PRICE_NEG, PriceSpec.PRICE_RND}:
        price_array, nth_firm_price = _share_correlated_number(
            _price_spec,
            share_uni_flag,
            _market_shares,
            nth_firm_share,
            _price_rng_seed_seq,
        )
    elif _price_spec.name.startswith("COST_"):
        _pcm_scaler, _pcm_scaler_nth = (
            (1.0, 1.0)
            if _price_spec == PriceSpec.COST_SYM
            else _share_correlated_number(
                _price_spec, False, _market_shares, nth_firm_share, _price_rng_seed_seq
            )
        )

        if share_uni_flag:
            _mktshr_plus = _market_shares
            _pcm_scaler_plus = _pcm_scaler
        else:
            _mktshr_plus = ArrayDouble(np.hstack((_market_shares, nth_firm_share)))
            _pcm_scaler_plus = (
                1.0
                if _price_spec == PriceSpec.COST_SYM
                else ArrayDouble(np.hstack((_pcm_scaler, _pcm_scaler_nth)))
            )

        pcm_array, mnl_test = _margins_sampler(
            _pcm_spec,
            _price_spec,
            _pcm_scaler_plus,
            _mktshr_plus,
            _aggr_purch_prob,
            _pcm_rng_seed_seq,
            _nthreads,
        )

        price_array = np.divide(_pcm_scaler_plus, 1 - pcm_array)
        if not share_uni_flag:
            price_array, nth_firm_price = price_array[:, :-1], price_array[:, [-1]]
            pcm_array = ArrayDouble(pcm_array[:, :-1])
    elif _price_spec != PriceSpec.PRICE_SYM:
        raise ValueError(
            f'Specification of price distribution, "{_price_spec.value}" is invalid.'
        )

    if not pcm_array.size:
        pcm_array, mnl_test = _margins_sampler(
            _pcm_spec,
            _price_spec,
            price_array,
            _market_shares,
            _aggr_purch_prob,
            _pcm_rng_seed_seq,
            _nthreads,
        )

    if _hsr_filing_test_type.name.startswith("HSR_"):
        # _mfra: computed merging firms' revenues over sample
        mfra = np.einsum("ij,ij->ij", price_array[:, :2], _market_shares[:, :2])

        if _hsr_filing_test_type == HSRFilingTest.SoP_TEN:
            hsr_filing_test = (
                np.divide((_s := np.sort(mfra, axis=1))[:, 1], _s[:, 0]) >= HSR_RATIO
            )

        elif _hsr_filing_test_type == HSRFilingTest.SoP_RND:
            hsr_test_share = np.empty_like(nth_firm_share)
            MultithreadedRNG(
                values=hsr_test_share,
                distribution="Beta",
                parameters=HSR_BETA_PARMS,
                seed_sequence=_hsr_rng_seed_seq,
                nthreads=_nthreads,
            ).fill()
            hsr_test_share = (
                _lower_bound + (1 - _lower_bound) * hsr_test_share
                if _lower_bound
                else hsr_test_share
            )
            hsr_test_rev = np.einsum("ij,ij->ij", nth_firm_price, hsr_test_share)
            hsr_filing_test = (
                np.sort(mfra, axis=1) / hsr_test_rev >= [1, HSR_RATIO]
            ).sum(axis=1) == mfra.shape[1]
            del hsr_test_share

        elif (
            _share_spec.distribution != SHRDistribution.UNI
            and _hsr_filing_test_type == HSRFilingTest.SoP_NTH
        ):
            # The nth firm test implemented here avoids an automatic 10-to-1 revenue ratio restriction. We implement as :
            # if the smaller merging firm matches or exceeds the n-th firm in size, and
            # the larger merging firm has at least 10 times the size of the nth firm,
            # the size test is considered met.
            # Alternatively, if the smaller merging firm has 10% or greater share,
            # # the value of transaction test is considered met.
            hsr_test_rev = np.einsum("ij,ij->ij", nth_firm_price, nth_firm_share)
            hsr_filing_test = (
                np.sort(mfra, axis=1) / hsr_test_rev >= [1, HSR_RATIO]
            ).sum(axis=1) == mfra.shape[1]
            del hsr_test_rev
        else:
            raise ValueError(
                f"Invalid value for _hsr_filing_test_type: {_hsr_filing_test_type} "
                f"with share distribution, {_share_spec.distribution}."
            )
        del mfra

    else:
        # Otherwise, all draws meet the filing test
        hsr_filing_test = np.full(len(_market_shares), True)

    return ArrayDouble(price_array), pcm_array, mnl_test, ArrayBoolean(hsr_filing_test)


def _share_correlated_number(
    _correlation_spec: PriceSpec,
    _share_uni_flag: bool,
    _market_shares: ArrayDouble,
    _nth_firm_share: ArrayDouble,
    _seed_sequence: SeedSequence | None,
) -> tuple[ArrayDouble, ArrayDouble]:
    nth_firm_price = EMPTY_ARRAYDOUBLE

    match _correlation_spec:
        case PriceSpec.PRICE_POS | PriceSpec.COST_POS:
            _pricing_array = _sharelator(_market_shares)
            if not _share_uni_flag:
                nth_firm_price = _sharelator(_nth_firm_share)

        case PriceSpec.PRICE_NEG | PriceSpec.COST_NEG:
            _pricing_array = _sharelator(1 - _market_shares)  # type: ignore
            if not _share_uni_flag:
                nth_firm_price = _sharelator(1 - _nth_firm_share)  # type: ignore

        case PriceSpec.PRICE_RND | PriceSpec.COST_RND:
            ncols = _market_shares.shape[1] + (0 if _share_uni_flag else 1)
            _pricing_array = (
                prng(_seed_sequence).integers(
                    1, 5 + 1, size=(len(_market_shares), ncols)
                )
                / 5.0
            )
            if not _share_uni_flag:
                nth_firm_price, _pricing_array = (
                    _pricing_array[:, -1:],
                    _pricing_array[:, :-1],
                )

        case _:
            raise ValueError(f"Invalid value here: {_correlation_spec}.")

    return ArrayDouble(_pricing_array), ArrayDouble(nth_firm_price)


def _sharelator(_market_shares: ArrayDouble) -> ArrayDouble:
    return ArrayDouble(np.ceil(_market_shares * 5.0) / 5.0)


def _margins_sampler(
    _pcm_spec: PCMSpec,
    _price_spec: PriceSpec,
    _pcm_scaler: float | ArrayDouble,
    _market_shares: ArrayDouble,  # mc if PriceSpec.COST_SYM else p
    _aggr_purch_prob: ArrayDouble,
    _pcm_rng_seed_seq: SeedSequence,
    _nthreads: int,
    /,
) -> tuple[ArrayDouble, ArrayBoolean]:
    _pcm_distribution, _pcm_parms, _pcm_restriction = (
        getattr(_pcm_spec, _f)
        for _f in ("distribution", "parameters", "pcm_restriction")
    )

    if _pcm_distribution in {PCMDistribution.EMPR_M, PCMDistribution.EMPR_U}:
        pcm_array = _margin_resampler(
            _pcm_distribution,
            _pcm_parms,
            sample_size=(
                (len(_market_shares), 1)
                if _pcm_spec.pcm_restriction in {PCMRestriction.SYM, PCMRestriction.MNL}
                else _market_shares.shape
            ),
            seed_sequence=_pcm_rng_seed_seq,
            nthreads=_nthreads,
        )

    else:
        pcm_array = ArrayDouble(
            np.empty_like(_market_shares[:, :1])
            if _pcm_spec.pcm_restriction in {PCMRestriction.SYM, PCMRestriction.MNL}
            else np.empty_like(_market_shares)
        )

        MultithreadedRNG(
            pcm_array,
            distribution=_pcm_distribution,
            parameters=_pcm_parms,
            seed_sequence=_pcm_rng_seed_seq,
            nthreads=_nthreads,
        ).fill()

    mnl_test = ArrayBoolean(np.full(len(pcm_array), True))
    if _pcm_restriction == PCMRestriction.SYM:
        pcm_array = ArrayDouble(np.hstack((pcm_array,) * _market_shares.shape[1]))
    elif _pcm_restriction == PCMRestriction.MNL:
        # Impose FOCs from profit-maximization with MNL demand
        purchase_prob_array = _aggr_purch_prob * _market_shares

        if _price_spec.name.startswith("COST_"):
            _nr_0 = np.divide(
                _pcm_scaler
                * np.einsum(
                    "ij,ij->ij", pcm_array[:, :1], 1 - purchase_prob_array[:, :1]
                )
                if isinstance(_pcm_scaler, int | float)
                else np.einsum(
                    "ij,ij,ij->ij",
                    _pcm_scaler[:, :1],
                    pcm_array[:, :1],
                    1 - purchase_prob_array[:, :1],
                ),
                1 - pcm_array[:, :1],
            )
            _dr_1 = (
                _pcm_scaler * (1 - purchase_prob_array[:, 1:])
                if isinstance(_pcm_scaler, int | float)
                else np.einsum(
                    "ij,ij->ij", _pcm_scaler[:, 1:], 1 - purchase_prob_array[:, 1:]
                )
            )
            pcm_array = ArrayDouble(
                np.hstack((pcm_array, np.divide(_nr_0, _nr_0 + _dr_1)))
            )
            del _nr_0, _dr_1

            return pcm_array, mnl_test

        else:
            pcm_array = ArrayDouble(
                np.hstack((
                    pcm_array,
                    np.divide(
                        np.einsum(
                            "ij,ij->ij",
                            pcm_array[:, :1],
                            1 - purchase_prob_array[:, :1],
                        ),
                        1 - purchase_prob_array[:, 1:],
                    )
                    if isinstance(_pcm_scaler, int | float)
                    else np.divide(
                        np.einsum(
                            "ij,ij,ij->ij",
                            _pcm_scaler[:, :1],
                            pcm_array[:, :1],
                            1 - purchase_prob_array[:, :1],
                        ),
                        np.einsum(
                            "ij,ij->ij",
                            _pcm_scaler[:, 1:],
                            1 - purchase_prob_array[:, 1:],
                        ),
                    ),
                ))
            )

        mnl_test = ArrayBoolean(
            np.logical_and(
                (pcm_array[:, 1:] >= 0).all(axis=1), (pcm_array[:, 1:] <= 1).all(axis=1)
            )
        )

    # else:  # This is a no-op, so commented out

    return pcm_array, mnl_test


def _margin_resampler(
    _dist_type: Literal[PCMDistribution.EMPR_M, PCMDistribution.EMPR_U],
    _dist_parms: EmpiricalMarginData,
    /,
    *,
    sample_size: tuple[int, int],
    seed_sequence: SeedSequence,
    nthreads: int,
) -> ArrayDouble:
    """Generate draws from the empirical distribution.

    Parameters
    ----------
    _dist_type
        Whether multi-modal (:attr:`PCMDistribution.EMPR_M`) or
        unimodal (:attr:`PCMDistribution.EMPR_U`)

    _dist_parms
        Margin vector and corresponding weights, kernel bandwidth estimate,
        and range of margins

    sample_size
        Size of generated array

    seed_sequence
        SeedSequence for seeding random-number generator when results
        are to be repeatable

    nthreads
        Number of threads to use in generating margin data.

    Returns
    -------
        Array of margin values

    """
    if _dist_type == PCMDistribution.EMPR_M:
        _sseq1, _sseq2 = seed_sequence.spawn(2)

        # We sample in parallel by column so that each thread generates
        # only SUBSAMPLE_SIZE floats, which limits memory consumption
        # over a larger number of threads.
        return ArrayDouble(
            _margin_resampler_multimodal_multithreaded(
                _dist_parms, sample_size, nthreads, _sseq1, _sseq2
            )
        )
    else:
        _alpha, _beta, _loc, _scale = beta_located_bound(_dist_parms.margin_stats)
        pcm_array = np.empty(sample_size)
        MultithreadedRNG(
            pcm_array,
            distribution="Beta",
            parameters=ArrayFloat([_alpha, _beta]),
            seed_sequence=seed_sequence,
            nthreads=nthreads,
        ).fill()
        return ArrayDouble(_loc + _scale * pcm_array)


def _margin_resampler_multimodal_multithreaded(
    _parms: EmpiricalMarginData,
    _ssz: tuple[int, int],
    _nthreads: int,
    _sseq1: SeedSequence,
    _sseq2: SeedSequence,
    /,
) -> NDArray[np.float64]:
    """Generate multithreaded draws on the empirical margin distribution."""
    _margin_values = np.repeat(
        _parms.average_gross_margins, (_parms.firm_counts * 10**3).astype(int)
    )
    _kernel_bandwidth = _parms.bandwidth
    _ssz_r, _ssz_c = _ssz
    _iter_count = ceil(_ssz_r / SUBSAMPLE_SIZE)
    _trunc_flag = _iter_count * SUBSAMPLE_SIZE > _ssz_r
    _trunc_size = _ssz_r % SUBSAMPLE_SIZE if _trunc_flag else SUBSAMPLE_SIZE

    return np.vstack(
        Parallel(backend="threading", n_jobs=min(_nthreads, _iter_count))(
            delayed(_multimodal_resampler)(
                _margin_values,
                _kernel_bandwidth,
                (
                    _trunc_size
                    if (1 + _ssq1i.spawn_key[-1]) == _iter_count
                    else SUBSAMPLE_SIZE,
                    _ssz_c,
                ),
                prng(_ssq1i),
                prng(_ssq2i),
            )
            for _ssq1i, _ssq2i in zip(
                *[_s.spawn(_iter_count) for _s in (_sseq1, _sseq2)], strict=True
            )
        )
    )


def _multimodal_resampler(
    _values: ArrayDouble,
    _bandwidth: ArrayDouble,
    _ssz: int | tuple[int, ...],
    _r1: Generator,
    _r2: Generator,
    /,
) -> NDArray[np.float64]:
    """Generate multimodal draws on the empirical (margin) distribution. [#_multimodal_resampler]_

    Parameters
    ----------
    _values
        Margin values

    _bandwidth
        Kernel bandwidth

    _ssz
        Size of generated array

    _r1, _r2
        Random number generators

    Returns
    -------
        Array of margin values


    References
    ----------
    .. [#_multimodal_resampler] See, https://kdepy.readthedocs.io/en/latest/examples.html#resampling-from-the-distribution

    """  # noqa: D400
    return _values[
        _r1.integers(len(_values), size=_ssz)
    ] + _bandwidth * _r2.standard_normal(size=_ssz)


def _beta_located(_mu: float, _sigma: float, /) -> ArrayFloat:
    """
    Given mean and stddev, return shape parameters (α, β) for corresponding Beta distribution.

    Solve the first two moments of the standard Beta to get the shape parameters.

    Parameters
    ----------
    _mu
        mean
    _sigma
        standard deviation

    Returns
    -------
        shape parameters for Beta distribution

    """  # noqa: RUF002
    mul = -1 + _mu * (1 - _mu) / (_sigma**2)
    return ArrayFloat([_mu * mul, (1 - _mu) * mul])


def beta_located_bound(
    _dist_parms: ArrayDouble | ArrayFloat, /, *, frac: float = 0.05
) -> ArrayFloat:
    R"""
    Return shape parameters (α, β), :math:`location`, and :math:`scale` for a non-standard beta, given mean, stddev, and range.

    Note, however, that we slightly expand the range here to
    :math:`\left[\max, \min\right] = \left[\mathrm{floor}(\min / 0.1) * 0.1, \mathrm{ceil}(\max / 0.1 ) * 0.1\right]`.
    Random variates are generated as :math:`location + scale \cdot \symup{Β}(α, β)`. [#beta]_

    Parameters
    ----------
    _dist_parms
        vector of :math:`\mu`, :math:`\sigma`, :math:`\min`, and :math:`\max` values

    Returns
    -------
        shape parameters for Beta distribution

    Notes
    -----
    For example, ``beta_located_bound(np.array([0.5, 0.2, 0.1, 0.9]))``. Note, with a high variance
    (:math:`\sigma^2`) or tight range, the relative frequency of extreme values exceeds those in
    the middle of the range, which is atypical of observed margins. A possible workaround is to set the
    min and max parameters to :code:`np.nan`, so that the range of margins is determined by
    the (shape paramters derived from) the mean and standard deviation.

    References
    ----------
    .. [#beta] NIST, Beta Distribution. https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    """  # noqa: RUF002
    _bmu, _bsigma, bmin, bmax = _dist_parms

    bmin = bmin if np.isnan(bmin) else np.floor(bmin / frac) * frac
    bmax = bmax if np.isnan(bmax) else np.ceil(bmax / frac) * frac
    bscale = bmax - bmin
    # return 4-parameter calibration: α, β, loc, scale  # noqa: RUF003
    return ArrayFloat(
        (*_beta_located(_bmu, _bsigma), 0, 1)
        if np.isnan(_dist_parms[2:]).any or np.array_equal(_dist_parms[2:], [0, 1])
        else (*_beta_located((_bmu - bmin) / bscale, _bsigma / bscale), bmin, bscale)
    )
