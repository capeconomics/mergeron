"""Methods to compute intrinsic enforcement/clearance rates from generated market data."""

from math import ceil

import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import parallel_config

from .. import EMPTY_ARRAYBIGINT
from .. import EMPTY_ARRAYDOUBLE
from .. import NTHREADS
from .. import VERSION
from .. import ArrayBIGINT
from .. import ArrayDouble
from .. import ArrayUINT8
from .. import UPPAggregator
from ..core import MGThresholds
from . import SUBSAMPLE_SIZE
from . import INVResolution
from . import MarketsData
from . import ShareSpec
from . import SHRDistribution
from . import UPPTestRegime
from . import UPPTestsCounts
from . import enforcement_stats as esl
from .data_generation_functions import compute_merging_firm_diversion_ratios
from .enforcement_stats import StatsGroup
from .enforcement_stats import compute_enforcement_counts

__version__ = VERSION


def compute_upp_test_counts(
    _share_spec: ShareSpec,
    _market_data_sample: MarketsData,
    _upp_test_parms: MGThresholds,
    _upp_test_regime: UPPTestRegime,
    /,
) -> UPPTestsCounts:
    """Estimate enforcement and clearance counts from market data sample.

    Parameters
    ----------
    _market_data_sample
        Market data sample

    _upp_test_parms
        Threshold values for various Guidelines criteria

    _upp_test_regime
        Specifies whether to analyze enforcement, clearance, or both
        and the GUPPI and diversion ratio aggregators employed, with
        default being to analyze enforcement based on the maximum
        merging-firm GUPPI and maximum diversion ratio between the
        merging firms

    Returns
    -------
    UPPTestsCounts
        Enforced and cleared counts

    """
    g_bar_, divr_bar_, cmcr_bar_, ipr_bar_ = (
        getattr(_upp_test_parms, _f) for _f in ("guppi", "dr", "cmcr", "ipr")
    )

    _frmshr_array, _pcm_array, _price_array = (
        getattr(_market_data_sample, _f)[:, :2]
        for _f in ("share_array", "pcm_array", "price_array")
    )
    _aggr_purch_prob = _market_data_sample.aggregate_purchase_probability
    _divratio_array = compute_merging_firm_diversion_ratios(
        _share_spec.recapture_form,
        _share_spec.recapture_rate,
        _frmshr_array,
        _aggr_purch_prob,
    )

    _share_array = _market_data_sample.share_array
    _hhi_delta = ArrayDouble(
        np.einsum("ij,ij->i", _frmshr_array, _frmshr_array[:, ::-1])[:, None]
    )
    if _share_spec.distribution == SHRDistribution.UNI:
        _fcounts, _hhi_post = ArrayUINT8(EMPTY_ARRAYBIGINT), EMPTY_ARRAYDOUBLE
    else:
        _fcounts = ArrayUINT8(
            np.einsum("ij->i", _share_array > 0, dtype="<u1")[:, None]
        )
        _hhi_post = ArrayDouble(
            _hhi_delta + np.einsum("ij,ij->i", _share_array, _share_array)[:, None]
        )

    _sample_size = len(_frmshr_array)
    _iter_count = ceil(_sample_size / SUBSAMPLE_SIZE)
    if _iter_count == 1:
        return _upp_test_counts(
            _divratio_array,
            _frmshr_array,
            _pcm_array,
            _price_array,
            _fcounts,
            _hhi_delta,
            _hhi_post,
            _upp_test_regime,
            g_bar_,
            divr_bar_,
            cmcr_bar_,
            ipr_bar_,
        )

    # Parallelize MNL test computation:
    with parallel_config(
        backend="threading", n_jobs=min(NTHREADS, _iter_count), return_as="generator"
    ):
        _res_list = Parallel()(
            delayed(_upp_test_counts)(
                _divratio_array[
                    (_si := _idx * SUBSAMPLE_SIZE) : (
                        _ei := (
                            len(_divratio_array)
                            if (_idx + 1) == _iter_count
                            else (_idx + 1) * SUBSAMPLE_SIZE
                        )
                    )
                ],
                _frmshr_array[_si:_ei],
                _pcm_array[_si:_ei],
                _price_array[_si:_ei],
                _fcounts[_si:_ei],
                _hhi_delta[_si:_ei],
                _hhi_post[_si:_ei],
                _upp_test_regime,
                g_bar_,
                divr_bar_,
                cmcr_bar_,
                ipr_bar_,
            )
            for _idx in range(_iter_count)
        )

    _res_list_stacks = [
        ArrayBIGINT(np.vstack([getattr(_j, _k) for _j in _res_list]))
        for _k in (f"{StatsGroup.FC}", f"{StatsGroup.DL}", f"{StatsGroup.HD}")
    ]
    del _res_list

    return UPPTestsCounts(*[
        (ArrayBIGINT([]) if not _g.any() else compute_enforcement_counts(_g, _h))
        for _g, _h in zip(
            _res_list_stacks, (StatsGroup.FC, StatsGroup.DL, StatsGroup.HD), strict=True
        )
    ])


def _upp_test_counts(
    _divratio_array: ArrayDouble,
    _frmshr_array: ArrayDouble,
    _pcm_array: ArrayDouble,
    _price_array: ArrayDouble,
    _fcounts: ArrayUINT8,
    _hhi_delta: ArrayDouble,
    _hhi_post: ArrayDouble,
    _upp_test_regime: UPPTestRegime,
    _g_bar: float,
    _divr_bar: float,
    _cmcr_bar: float,
    _ipr_bar: float,
    /,
) -> UPPTestsCounts:
    guppi_array, ipr_array, cmcr_array = (
        ArrayDouble(np.empty_like(_divratio_array)) for _ in range(3)
    )

    np.einsum(
        "ij,ij,ij->ij",
        _divratio_array,
        _pcm_array[:, ::-1],
        _price_array[:, ::-1] / _price_array,
        out=guppi_array,
    )

    np.divide(
        np.einsum("ij,ij->ij", _pcm_array, _divratio_array),
        1 - _divratio_array,
        out=ipr_array,
    )

    np.divide(ipr_array, 1 - _pcm_array, out=cmcr_array)

    (divr_test_vector,) = _compute_test_vector_seq(
        (_divratio_array,), _frmshr_array, _upp_test_regime.diversion_aggregator
    )

    (guppi_test_vector, cmcr_test_vector, ipr_test_vector) = _compute_test_vector_seq(
        (guppi_array, cmcr_array, ipr_array),
        _frmshr_array,
        _upp_test_regime.guppi_aggregator,
    )
    del cmcr_array, ipr_array, guppi_array

    if _upp_test_regime.resolution == INVResolution.ENFT:
        upp_test_arrays = np.hstack((
            guppi_test_vector >= _g_bar,
            (guppi_test_vector >= _g_bar) | (divr_test_vector >= _divr_bar),
            cmcr_test_vector >= _cmcr_bar,
            ipr_test_vector >= _ipr_bar,
        ))
    else:
        upp_test_arrays = np.hstack((
            guppi_test_vector < _g_bar,
            (guppi_test_vector < _g_bar) & (divr_test_vector < _divr_bar),
            cmcr_test_vector < _cmcr_bar,
            ipr_test_vector < _ipr_bar,
        ))

    # Clearance counts by firm count
    enf_cnts_sim_byfirmcount_array = (
        compute_enforcement_counts(
            ArrayBIGINT(
                np.hstack((_fcounts, np.ones_like(_fcounts, int), upp_test_arrays))
            ),
            StatsGroup.FC,
        )
        if _fcounts.size
        else EMPTY_ARRAYBIGINT
    )

    # Clearance counts by ΔHHI
    enf_cnts_sim_bydelta_array = compute_enforcement_counts(
        ArrayBIGINT(
            np.hstack((
                ArrayBIGINT(esl.hhi_delta_ranger(_hhi_delta)),
                np.ones_like(_hhi_delta, int),
                upp_test_arrays,
            ))
        ),
        StatsGroup.DL,
    )

    # Clearance counts by post-merger HHI
    enf_cnts_sim_byhhianddelta_array = (
        EMPTY_ARRAYBIGINT
        if (not _hhi_post.size) or np.isnan(next(_hhi_post.flat))
        else compute_enforcement_counts(
            ArrayBIGINT(
                np.hstack((
                    ArrayBIGINT(esl.hhi_post_ranger(_hhi_post)),
                    ArrayBIGINT(esl.hhi_delta_ranger(_hhi_delta)),
                    np.ones_like(_hhi_post, int),
                    upp_test_arrays,
                ))
            ),
            StatsGroup.HD,
        )
    )

    return UPPTestsCounts(
        ByFirmCount=enf_cnts_sim_byfirmcount_array,
        ByDelta=enf_cnts_sim_bydelta_array,
        ByHHIandDelta=enf_cnts_sim_byhhianddelta_array,
    )


def _compute_test_vector_seq(
    _test_data_seq: tuple[ArrayDouble, ...],
    _weights: ArrayDouble,
    _aggregator: UPPAggregator,
) -> tuple[ArrayDouble, ...]:
    if _aggregator in {UPPAggregator.CPA, UPPAggregator.CPD, UPPAggregator.CPG}:
        _wgts = _weights[:, ::-1] / np.einsum("ij->i", _weights)[:, None]
    elif _aggregator in {UPPAggregator.OSA, UPPAggregator.OSD, UPPAggregator.OSG}:
        _wgts = _weights / np.einsum("ij->i", _weights)[:, None]
    elif _aggregator in {UPPAggregator.AVG, UPPAggregator.DIS, UPPAggregator.GMN}:
        _wgts = (_w := np.ones((1, 2))) / _w.sum()
    else:
        # Weights not used for min, max
        _wgts = np.array([], float)

    # Use weights calculated above to compute average, distance, and geometric mean:
    if _aggregator in {UPPAggregator.AVG, UPPAggregator.CPA, UPPAggregator.OSA}:
        test_vector_seq = (
            np.einsum("ij,ij->i", _wgts, _g)[:, None] for _g in _test_data_seq
        )
    elif _aggregator in {UPPAggregator.DIS, UPPAggregator.CPD, UPPAggregator.OSD}:
        test_vector_seq = (
            np.sqrt(np.einsum("ij,ij,ij->i", _wgts, _g, _g))[:, None]
            for _g in _test_data_seq
        )
    elif _aggregator in {UPPAggregator.CPG, UPPAggregator.GMN, UPPAggregator.OSG}:
        test_vector_seq = (
            np.expm1(np.einsum("ij,ij->i", _wgts, np.log1p(_g)))[:, None]
            for _g in _test_data_seq
        )
    elif _aggregator == UPPAggregator.MAX:
        test_vector_seq = (_g.max(axis=1, keepdims=True) for _g in _test_data_seq)
    elif _aggregator == UPPAggregator.MIN:
        test_vector_seq = (_g.min(axis=1, keepdims=True) for _g in _test_data_seq)
    else:
        raise ValueError("GUPPI/diversion ratio aggregation method is invalid.")
    return tuple(ArrayDouble(_t) for _t in test_vector_seq)


if __name__ == "__main__":
    print(
        "This module defines functions for generating UPP test arrays "
        "and UPP test enforcement-counts on market samples."
    )
