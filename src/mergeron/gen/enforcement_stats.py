"""Methods to format and print summary statistics on merger enforcement patterns."""

import enum
from collections.abc import Mapping
from typing import Literal

import numpy as np
from scipy.interpolate import make_interp_spline

from .. import EMPTY_ARRAYBIGINT
from .. import VERSION
from .. import YAML
from .. import ArrayBIGINT
from .. import ArrayUINT8
from .. import Enameled
from ..core import DELTA_HEADER_DICT
from ..core import HHI_HEADER_DICT
from ..core import INVData
from ..core import INVTableData
from . import INVResolution
from . import StatsGroup

__version__ = VERSION


@YAML.register_class
@enum.unique
class IndustryGroup(str, Enameled):
    """Industry group of reported markets."""

    ALL = "All Markets"
    GRO = "Grocery Markets"
    OIL = "Oil Markets"
    CHM = "Chemical Markets"
    PHM = "Pharmaceuticals Markets"
    HOS = "Hospital Markets"
    EDS = "Electronically-Controlled Devices and Systems Markets"
    BRD = "Branded Consumer Goods Markets"
    OTH = '"Other" Markets'
    IIC = "Industries in Common"


@YAML.register_class
@enum.unique
class OtherEvidence(str, Enameled):
    """Additional evidence available, if any, for reported markets."""

    HOT = "Hot Documents Identified"
    NHT = "No Hot Documents Identified"
    HTU = "No Evidence on Hot Documents"
    NCC = "No Strong Customer Complaints"
    SCC = "Strong Customer Complaints"
    CCU = "No Evidence on Customer Complaints"
    END = "Entry Difficult"
    EEY = "Entry Easy"
    EEU = "No Entry Evidence"
    UNR = "Unrestricted on additional evidence"


# Parameters and functions to interpolate selected HHI and ΔHHI values
#   recorded in fractions to ranges of values in points on the HHI scale
HHI_POST_KNOTS = np.array(
    [*[_h for _h in HHI_HEADER_DICT.values() if _h < 10001], 10001], int
)
HHI_DELTA_KNOTS = np.array(
    [*[_d for _d in DELTA_HEADER_DICT.values() if _d < 5001], 5001], int
)
hhi_post_ranger, hhi_delta_ranger = (
    make_interp_spline(_f / 1e4, _f, k=0) for _f in (HHI_POST_KNOTS, HHI_DELTA_KNOTS)
)


def enforcement_counts_observed(
    _invdata_array_dict: INVData,
    _data_period: str,
    _table_industry_group: IndustryGroup,
    _table_other_evidence: OtherEvidence,
    _stats_group: Literal[StatsGroup.FC, StatsGroup.DL, StatsGroup.HD],
    _enf_spec: INVResolution,
    /,
) -> ArrayBIGINT:
    """Summarize investigations data by reporting group.

    Parameters
    ----------
    _invdata_array_dict
        raw investigations data
    _data_period
        investigations data reporting-period
    _table_industry_group
        industry group
    _table_other_evidence
        additional evidence
    _stats_group
        grouping measure
    _enf_spec
        enforcement specification (see, :class:`mergeron.gen.INVResolution`)

    Returns
    -------
    ArrayBIGINT
        Counts of markets resolved as enforced, cleared, or both, respectively.
    """
    if _data_period not in _invdata_array_dict:
        raise ValueError(
            f"Invalid value of data period, {f'"{_data_period}"'}."
            f"Must be one of, {tuple(_invdata_array_dict.keys())!r}."
        )

    _ndim_in = 2 if _stats_group == StatsGroup.HD else 1
    _table_type = (
        _stats_group.value if _stats_group == StatsGroup.FC else StatsGroup.HD.value
    )

    _data_array_dict_sub = _invdata_array_dict[_data_period][_table_type]

    _table_no = table_no_lku(
        _data_array_dict_sub, _table_industry_group, _table_other_evidence
    )

    _data_array = _data_array_dict_sub[_table_no].data_array

    stats_kept_indxs = []
    match _enf_spec:
        case INVResolution.CLRN:
            stats_kept_indxs = [-1, -2]
        case INVResolution.ENFT:
            stats_kept_indxs = [-1, -3]
        case INVResolution.BOTH:
            stats_kept_indxs = [-1, -3, -2]

    _counts_array = ArrayBIGINT(
        np.hstack([
            _data_array[:, [1]]
            if _stats_group == StatsGroup.DL
            else _data_array[:, :_ndim_in],
            _data_array[:, stats_kept_indxs],
        ])
    )

    return enforcement_counts(_counts_array, _stats_group)


def table_no_lku(
    _data_array_dict_sub: Mapping[str, INVTableData],
    _table_ind_group: IndustryGroup = IndustryGroup.ALL,
    _table_evid_cond: OtherEvidence = OtherEvidence.UNR,
    /,
) -> str:
    """Lookup table number based on industry group and additional evidence."""
    if _table_evid_cond not in (
        _egl := [
            _data_array_dict_sub[_v].additional_evidence for _v in _data_array_dict_sub
        ]
    ):
        raise ValueError(
            f"Invalid value for additional evidence, {f'"{_table_evid_cond}"'}."
            f"Must be one of {_egl!r}"
        )
    if _table_ind_group not in (
        _igl := [_data_array_dict_sub[_v].industry_group for _v in _data_array_dict_sub]
    ):
        raise ValueError(
            f"Invalid value for industry group, {f'"{_table_ind_group}"'}."
            f"Must be one of {_igl!r}"
        )

    return next(
        _t
        for _t in _data_array_dict_sub
        if all((
            _data_array_dict_sub[_t].industry_group == _table_ind_group,
            _data_array_dict_sub[_t].additional_evidence == _table_evid_cond,
        ))
    )


def enforcement_counts(
    _raw_counts: ArrayBIGINT | ArrayUINT8, _stats_group: StatsGroup, /
) -> ArrayBIGINT:
    """Summarize investigations data.

    Parameters
    ----------
    _raw_counts
        raw investigations data array

    _stats_group
        measure for grouping enforcement counts - firm count, post-mergerHHI,
        or ΔHHI

    Returns
    -------
    ArrayBIGINT
        Enforcement counts by range/value of firm count, post-merger HHI, or ΔHHI
    """
    if (not _raw_counts.size) or np.isnan(next(_raw_counts.flat)):
        return EMPTY_ARRAYBIGINT
    elif _stats_group == StatsGroup.FC:
        return ArrayBIGINT(
            np.vstack([
                np.concatenate([
                    (_i,),
                    np.einsum(
                        "ij->j", _raw_counts[_raw_counts[:, 0] == _i][:, 1:], dtype=int
                    ),
                ])
                for _i in np.unique(_raw_counts[:, 0])
            ])
        )
    elif _stats_group == StatsGroup.DL:
        return ArrayBIGINT(
            np.vstack([
                np.concatenate([
                    (_i,),
                    np.einsum(
                        "ij->j", _raw_counts[_raw_counts[:, 0] == _i][:, 1:], dtype=int
                    ),
                ])
                for _i in HHI_DELTA_KNOTS[:-1]
            ])
        )
    elif _stats_group == StatsGroup.HD:
        ret_val = ArrayBIGINT(np.zeros((1, _raw_counts.shape[1]), dtype=int))
        # rollup clearance stats by HHI and Delta thresholds
        for _hhi_post_lim in HHI_POST_KNOTS[:-1]:
            _raw_counts_i = _raw_counts[_raw_counts[:, 0] == _hhi_post_lim]

            for _delta_lim in HHI_DELTA_KNOTS[:-1]:
                _raw_counts_ij = _raw_counts_i[_raw_counts_i[:, 1] == _delta_lim]

                ret_val = np.vstack((
                    ret_val,
                    np.asarray(
                        [
                            _hhi_post_lim,
                            _delta_lim,
                            *np.einsum("ij->j", _raw_counts_ij[:, 2:], dtype=int),
                        ],
                        dtype=int,
                    ),
                ))
        return ArrayBIGINT(ret_val[1:])
    else:
        raise NotImplementedError(f"Unsupported stats group, {_stats_group!r}")
