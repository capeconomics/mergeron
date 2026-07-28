"""Methods to load and augment FTC Merger Investigations Data.

Details on downloading and processing the data are provided in
the module, :code:`mergeron.core.process_ftc_merger_investigations_data`.

Notes
-----
Reported row and column totals from source data are not stored.

"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from .. import EMPTY_ARRAYBIGINT
from .. import VERSION
from .. import YAML
from .. import _dict_from_mapping
from .. import _mappingproxy_from_mapping
from .. import zipfile
from . import FCOUNT_TABLE_ALL
from . import HHI_TABLE_ALL
from . import INVDATA_ARCHIVE_PATH
from . import TABLE_TYPES
from . import INVData
from . import INVDataDict
from . import INVTableData
from .process_ftc_merger_investigations_data import get_table_type
from .process_ftc_merger_investigations_data import parse_ftc_reports

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__version__ = VERSION

# cspell: "includeRegExpList": ["strings", "comments", /( {3}['"]{3}).*?\\1/g]


def construct_data(
    _archive_path: Path = INVDATA_ARCHIVE_PATH,
    *,
    flag_backward_compatibility: bool = True,
    flag_pharma_for_exclusion: bool = True,
    rebuild_data: bool = False,
) -> INVData:
    """Construct FTC merger investigations data for added non-overlapping periods.

    FTC merger investigations data are reported in cumulative periods,
    e.g., 1996-2003 and 1996-2011, but the analyst may want data reported in
    non-overlapping periods, e.g., 2004-2011. Given the way in which FTC had
    reported merger investigations data, the above example is the only instance
    in which the 1996-2003 data can be subtracted from the cumulative data to
    extract merger investigations data for the later period.
    See also, Kwoka, Sec. 2.3.3. [#]_

    Parameters
    ----------
    _archive_path
        Path to file container for serialized constructed data
    flag_backward_compatibility
        Flag whether the reported data should be treated as backward-compatible
    flag_pharma_for_exclusion
        Flag whether data for Pharmaceuticals is included in,  the set of
        industry groups with consistent reporting in both early and late periods

    Returns
    -------
        A dictionary of merger investigations data keyed to reporting periods

    References
    ----------

    .. [#] Kwoka, J., Greenfield, D., & Gu, C. (2015). Mergers, merger control,
       and remedies: A retrospective analysis of U.S. policy. MIT Press.

    """
    if _archive_path.is_file() and not rebuild_data:
        with (
            zipfile.ZipFile(_archive_path, "r") as _yzh,
            _yzh.open(f"{_archive_path.stem}.yaml", "r") as _yfh,
        ):
            invdata_: INVData = YAML.load(_yfh)
        return invdata_

    invdata: INVDataDict = _dict_from_mapping(parse_ftc_reports())

    # Add some data periods (
    #   only periods ending in 2011, others have few observations and
    #   some incompatibilities
    #   )
    for data_period in "2004-2011", "2006-2011", "2008-2011":
        _construct_new_period_data(
            invdata,
            data_period,
            flag_backward_compatibility=flag_backward_compatibility,
        )

    # Create data for industries with no evidence on entry
    for data_period in invdata:
        _construct_no_evidence_data(invdata, data_period)

    # Create a list of exclusions to named industries in the base period,
    #   for construction of aggregate enforcement statistics where feasible
    industry_exclusion_list = {
        "AllMarkets",
        "OtherMarkets",
        "IndustriesinCommon",
        "",
        ("PharmaceuticalsMarkets" if flag_pharma_for_exclusion else None),
    }

    # Construct aggregate tables
    for data_period in "1996-2003", "1996-2011", "2004-2011":
        invdata_sub_ = invdata[data_period]
        for table_no in (HHI_TABLE_ALL, FCOUNT_TABLE_ALL):
            _table_type = get_table_type(table_no)
            aggr_tables_list = [
                t_
                for t_ in invdata["1996-2003"]
                if invdata["1996-2003"][t_].table_type == _table_type
                and re.sub(r"\W", "", invdata["1996-2003"][t_].industry_group)
                not in industry_exclusion_list
            ]

            invdata_sub_ |= {
                table_no.replace(".1", ".X"): invdata_build_aggregate_table(
                    invdata_sub_, aggr_tables_list
                )
            }

    retval: INVData = _mappingproxy_from_mapping(invdata)
    with (
        zipfile.ZipFile(_archive_path, "w", compression=93) as _yzh,
        _yzh.open(f"{_archive_path.stem}.yaml", "w") as _yfh,
    ):
        YAML.dump(retval, _yfh)

    return retval


def _construct_no_evidence_data(_invdata: INVDataDict, _data_period: str, /) -> None:
    _ind_grp = "All Markets"
    _table_nos_map = dict(
        zip(
            (
                "No Entry Evidence",
                "No Evidence on Customer Complaints",
                "No Evidence on Hot Documents",
            ),
            (
                ("Table 9.X", "Table 10.X"),
                ("Table 7.X", "Table 8.X"),
                ("Table 5.X", "Table 6.X"),
            ),
            strict=True,
        )
    )
    for _evid_cond in (
        "No Entry Evidence",
        "No Evidence on Customer Complaints",
        "No Evidence on Hot Documents",
    ):
        for _dtn in _table_nos_map[_evid_cond]:
            invdata_sub_ = _invdata[_data_period]
            _table_type = get_table_type(_dtn)
            _stn0 = "Table 4.1" if _table_type == TABLE_TYPES[1] else "Table 3.1"
            _stn1, _stn2 = (_dtn.replace(".X", f".{_i}") for _i in ("1", "2"))

            invdata_sub_ |= {
                _dtn: INVTableData(
                    _table_type,
                    _ind_grp,
                    _evid_cond,
                    np.hstack((
                        invdata_sub_[_stn0].data_array[:, :-3],
                        (
                            invdata_sub_[_stn0].data_array[:, -3:]
                            - invdata_sub_[_stn1].data_array[:, -3:]
                            - invdata_sub_[_stn2].data_array[:, -3:]
                        ),
                    )),
                )
            }


def _construct_new_period_data(
    _invdata: INVDataDict,
    _data_period: str,
    /,
    *,
    flag_backward_compatibility: bool = False,
) -> None:
    cuml_period = f"1996-{_data_period.split('-')[1]}"
    if cuml_period != "1996-2011":
        raise ValueError('Expected cumulative period, "1996-2011"')

    invdata_cuml = _invdata[cuml_period]

    base_period = "1996-{}".format(int(_data_period.split("-", maxsplit=1)[0]) - 1)
    invdata_base = _invdata[base_period]

    invdata_build = {}
    for table_no in invdata_cuml:
        invdata_cuml_table = invdata_cuml[table_no]
        _table_type, _ind_group, _evid_cond, _cuml_array = (
            invdata_cuml_table.table_type,
            invdata_cuml_table.industry_group,
            invdata_cuml_table.additional_evidence,
            invdata_cuml_table.data_array,
        )

        invdata_base_table = invdata_base.get(
            table_no, INVTableData("", "", "", EMPTY_ARRAYBIGINT)
        )

        (_base_ind_group, _base_array) = (
            getattr(invdata_base_table, _a) for _a in ("industry_group", "data_array")
        )

        # Some tables can't be constructed due to inconsistencies in the data
        # across time periods
        if (
            (_data_period != "2004-2011" and _ind_group != "All Markets")
            or (_ind_group in {'"Other" Markets', "Industries in Common"})
            or (_base_ind_group in {'"Other" Markets', ""})
        ):
            continue

        # NOTE: Clean data to enforce consistency in FTC data
        if flag_backward_compatibility:
            # Consistency here means that the number of investigations reported
            # in each period is no less than the number reported in
            # any prior period. Although the time periods for table 3.2 through 3.5
            # are not the same in the data for 1996-2005 and 1996-2007 as in
            # the data for the other periods, they are nonetheless shorter than
            # the period 1996-2011, and hence the counts reported for 1996-2011
            # cannot be less than those reported in these prior periods. Note that
            # The number of "revisions" applied below, for enforcing consistency,
            # is sufficiently small as to be unlikely to substantially impact
            # results from analysis of the data.
            _cuml_array_stack = []
            _base_array_stack = []

            for data_period_detail in _invdata:
                pd_start, pd_end = (int(g) for g in data_period_detail.split("-"))
                if pd_start == 1996:
                    _cuml_array_stack += [
                        _invdata[data_period_detail][table_no].data_array[:, -3:-1]
                    ]
                if pd_start == 1996 and pd_end < int(
                    _data_period.split("-", maxsplit=1)[0]
                ):
                    _base_array_stack += [
                        _invdata[data_period_detail][table_no].data_array[:, -3:-1]
                    ]
            _cuml_array_enfcls, _base_array_enfcls = (
                np.stack(_f).max(axis=0)
                for _f in (_cuml_array_stack, _base_array_stack)
            )
            _bld_array_enfcls = _cuml_array_enfcls - _base_array_enfcls
        else:
            # Consistency here means that the most recent data are considered
            # the most accurate, and when constructing data for a new period
            # any negative counts for merger investigations "enforced" or "closed"
            # are reset to zero (non-negativity). The above convention is adopted
            # on the basis of discussions with FTC staff, and given that FTC does
            # not assert backward compatibility published data on
            # merger investigations. Also, FTC appears to maintain that
            # the most recently published data are considered the most accurate
            # account of the pattern of FTC investigations of horizontal mergers,
            # and that the figures for any reported period represent the most
            # accurate data for that period. The published data may not be fully
            # backward compatible due to minor variation in (applying) the criteria
            # for inclusion, as well as industry coding, undertaken to maintain
            # transparency on the enforcement process.
            _bld_array_enfcls = _cuml_array[:, -3:-1] - _base_array[:, -3:-1]

            # # // spellchecker: disable
            # To examine the number of corrected values per table,  // spellchecker: disable
            # uncomment the statements below
            # invdata_array_bld_tbc = where(
            #   invdata_array_bld_enfcls < 0, invdata_array_bld_enfcls, 0
            # )
            # if np.einsum('ij->', invdata_array_bld_tbc):
            #     print(
            #       f"{_data_period}, {_table_no}, {invdata_ind_group}:",
            #       abs(np.einsum('ij->', invdata_array_bld_tbc))
            #       )
            # #  // spellchecker: disable

            # Enforce non-negativity
            _bld_array_enfcls = np.stack((
                _bld_array_enfcls,
                np.zeros_like(_bld_array_enfcls),
            )).max(axis=0)

        invdata_array_bld = np.hstack((
            _cuml_array[:, :-3],
            _bld_array_enfcls,
            np.einsum("ij->i", _bld_array_enfcls)[:, None],
        ))

        invdata_build[table_no] = INVTableData(
            _table_type, _ind_group, _evid_cond, invdata_array_bld
        )
        del _table_type, _ind_group, _evid_cond, _cuml_array
        del _base_ind_group, _base_array
    _invdata[_data_period] = invdata_build


def invdata_build_aggregate_table(
    _data_periodsub: dict[str, INVTableData], _aggr_table_list: Sequence[str]
) -> INVTableData:
    """Aggregate selected FTC merger investigations data tables within a given time period."""
    hdr_table_no = _aggr_table_list[0]
    _table_type = get_table_type(hdr_table_no)

    return INVTableData(
        _table_type,
        "Industries in Common",
        "Unrestricted on additional evidence",
        np.hstack((
            _data_periodsub[hdr_table_no].data_array[:, :-3],
            np.einsum(
                "ijk->jk",
                np.stack([
                    (_data_periodsub[t_]).data_array[:, -3:] for t_ in _aggr_table_list
                ]),
            ),
        )),
    )


if __name__ == "__main__":
    print(
        "This module defines functions for downloading and preparing FTC merger investigations data for further analysis."
    )
