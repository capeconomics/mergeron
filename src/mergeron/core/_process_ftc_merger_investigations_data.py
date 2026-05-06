"""Download and parse FTC Merger Investigations Data.

This module provided as documentation only. The package
:code:`pymupdf` is a requirement of this module but is
distributed under a license that  may be incompatible with
the MIT license under which this package is distributed.

"""

import re
from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np

# import pymupdf
import urllib3
from bs4 import BeautifulSoup
from numpy.testing import assert_array_equal

from .. import ArrayBIGINT
from .. import _mappingproxy_from_mapping
from . import DELTA_HEADER_DICT
from . import FCOUNT_HEADER_DICT
from . import FID_WORK_DIR
from . import HHI_HEADER_DICT
from . import HHI_TABLE_ALL
from . import TABLE_TYPES
from . import TTL_KEY
from . import INVData
from . import INVDataDict
from . import INVTableData

TABLE_NO_RE = re.compile(r"Table \d+\.\d+")


# mypy: disable-error-code="unreachable"
def _parse_invdata() -> INVData:
    """Parse FTC merger investigations data reports to structured data.

    Returns
    -------
        Immutable dictionary of merger investigations data, keyed to
        reporting period, and including all tables organized by
        Firm Count (number of remaining competitors) and
        by range of HHI and ∆HHI.
    """
    raise ValueError(
        "This function is defined here as documentation.\n"
        "NOTE: License for `pymupdf`, upon which this function depends,"
        " may be incompatible with the MIT license,"
        " under which this pacakge is distributed."
        " Making this fumction operable requires the user to modify"
        " the source code as well as to install an additional package"
        " not distributed with this package or identified as a requirement."
    )

    invdata_docnames = _download_invdata(FID_WORK_DIR)

    invdata: INVDataDict = {}

    for invdata_docname in invdata_docnames:
        invdata_pdf_path = FID_WORK_DIR.joinpath(invdata_docname)

        invdata_doc = pymupdf.open(invdata_pdf_path)  # type: ignore  # noqa: F821
        invdata_meta = invdata_doc.metadata
        if invdata_meta["title"] == " ":
            invdata_meta["title"] = ", ".join((
                "Horizontal Merger Investigation Data",
                "Fiscal Years",
                "1996-2005",
            ))

        data_period = "".join(  # line-break here for readability
            re.findall(r"(\d{4}) *(-) *(\d{4})", invdata_meta["title"])[0]
        )

        # Initialize containers for parsed data
        invdata[data_period] = {k: {} for k in TABLE_TYPES}

        for pdf_pg in invdata_doc.pages():
            doc_pg_blocks = pdf_pg.get_text("blocks", sort=False)
            # Across all published reports of FTC investigations data,
            #   sorting lines (PDF page blocks) by the lower coordinates
            #   and then the left coordinates is most effective for
            #   ordering table rows in top-to-bottom order; this doesn't
            #   work for the 1996-2005 data, however, so we resort later
            doc_pg_blocks = sorted([
                (f"{_f[3]:03.0f}{_f[0]:03.0f}{_f[1]:03.0f}{_f[2]:03.0f}", *_f)
                for _f in doc_pg_blocks
                if _f[-1] == 0
            ])

            data_blocks: list[tuple[str]] = [("",)]
            # Pages layouts not the same in all reports
            pg_hdr_strings = (
                "FEDERAL TRADE COMMISSION",
                "HORIZONTAL MERGER INVESTIGATION DATA: FISCAL YEARS 1996 - 2011",
            )
            if len(doc_pg_blocks) > 4:
                tnum = None
                for _pg_blk in doc_pg_blocks:
                    if tnum := TABLE_NO_RE.fullmatch(_pg_blk[-3].strip()):
                        data_blocks = [
                            b_
                            for b_ in doc_pg_blocks
                            if not b_[-3].startswith(pg_hdr_strings)
                            and (
                                b_[-3].strip()
                                not in {"Significant Competitors", "Post Merger HHI"}
                            )
                            and not re.fullmatch(r"\d+", b_[-3].strip())
                        ]
                        break
                if not tnum:
                    continue
                del tnum
            else:
                continue

            _parse_page_blocks(invdata, data_period, data_blocks)

        invdata_doc.close()

    return _mappingproxy_from_mapping(invdata)


def _parse_page_blocks(
    _invdata: INVDataDict, _data_period: str, _doc_pg_blocks: Sequence[Sequence[Any]], /
) -> None:
    if _data_period != "1996-2011":
        _parse_table_blocks(_invdata, _data_period, _doc_pg_blocks)
    else:
        test_list = [
            (g, f[-3].strip())
            for g, f in enumerate(_doc_pg_blocks)
            if TABLE_NO_RE.fullmatch(f[-3].strip())
        ]
        # In the 1996-2011 report, there are 2 tables per page
        if len(test_list) == 1:
            table_a_blocks = _doc_pg_blocks
            table_b_blocks: Sequence[Sequence[Any]] = []
        else:
            table_a_blocks, table_b_blocks = (
                _doc_pg_blocks[test_list[0][0] : test_list[1][0]],
                _doc_pg_blocks[test_list[1][0] :],
            )

        for table_i_blocks in table_a_blocks, table_b_blocks:
            if not table_i_blocks:
                continue
            _parse_table_blocks(_invdata, _data_period, table_i_blocks)


def _parse_table_blocks(
    _invdata: INVDataDict, _data_period: str, _table_blocks: Sequence[Sequence[str]], /
) -> None:
    invdata_evid_cond = "Unrestricted on additional evidence"
    table_num, table_ser, table_type = _identify_table_type(
        _table_blocks[0][-3].strip()
    )

    if _data_period == "1996-2011":
        invdata_ind_group = (
            _table_blocks[1][-3].split("\n")[1]
            if table_num == "Table 4.8"
            else _table_blocks[2][-3].split("\n", maxsplit=1)[0]
        )

        if table_ser > 4:
            invdata_evid_cond = (
                _table_blocks[2][-3].split("\n")[1]
                if table_ser in {9, 10}
                else _table_blocks[3][-3].strip()
            )

    elif _data_period == "1996-2005":
        _table_blocks = sorted(_table_blocks, key=itemgetter(6))

        invdata_ind_group = _table_blocks[3][-3].strip()
        if table_ser > 4:
            invdata_evid_cond = _table_blocks[5][-3].strip()

    elif table_ser % 2 == 0:
        invdata_ind_group = _table_blocks[1][-3].split("\n")[2]
        if (evid_cond_teststr := _table_blocks[2][-3].strip()) == "Outcome":
            invdata_evid_cond = "Unrestricted on additional evidence"
        else:
            invdata_evid_cond = evid_cond_teststr

    elif _table_blocks[3][-3].startswith("FTC Horizontal Merger Investigations"):
        invdata_ind_group = _table_blocks[3][-3].split("\n")[2]
        invdata_evid_cond = "Unrestricted on additional evidence"

    else:
        # print(_table_blocks)
        invdata_evid_cond = (
            _table_blocks[1][-3].strip()
            if table_ser == 9
            else _table_blocks[3][-3].strip()
        )
        invdata_ind_group = _table_blocks[4][-3].split("\n")[2]

    if invdata_ind_group == "Pharmaceutical Markets":
        invdata_ind_group = "Pharmaceuticals Markets"

    process_table_func = (
        _process_table_blks_conc_type
        if table_type == TABLE_TYPES[0]
        else _process_table_blks_cnt_type
    )

    table_array = process_table_func(_table_blocks)
    if not isinstance(table_array, np.ndarray) or table_array.dtype != int:
        print(table_num)
        print(_table_blocks)
        raise ValueError

    table_data = INVTableData(invdata_ind_group, invdata_evid_cond, table_array)
    _invdata[_data_period][table_type] |= {table_num: table_data}


def _identify_table_type(_tnstr: str = HHI_TABLE_ALL, /) -> tuple[str, int, str]:
    tnum = _tnstr.split(" ")[1]
    tsub = int(tnum.split(".")[0])
    return _tnstr, tsub, TABLE_TYPES[(tsub + 1) % 2]


def _process_table_blks_conc_type(
    _table_blocks: Sequence[Sequence[str]], /
) -> ArrayBIGINT:
    conc_row_pat = re.compile(r"((?:0|\d,\d{3}) (?:- \d+,\d{3}|\+)|TOTAL)")

    col_titles = tuple(DELTA_HEADER_DICT.values())
    col_totals = np.zeros(len(col_titles), int)
    invdata_array = ArrayBIGINT(np.array([], int))

    for tbl_blk in _table_blocks:
        if conc_row_pat.match(_blk_str := tbl_blk[-3]):
            row_list: list[str] = _blk_str.strip().split("\n")
            row_title: str = row_list.pop(0)
            row_key: int = (
                7000 if row_title.startswith("7,000") else HHI_HEADER_DICT[row_title]
            )
            row_total = np.array(row_list.pop().replace(",", "").split("/"), int)
            data_row_list: list[list[int]] = []
            while row_list:
                enfd_val, clsd_val = row_list.pop(0).split("/")
                data_row_list += [
                    [
                        row_key,
                        col_titles[len(data_row_list)],
                        int(enfd_val),
                        int(clsd_val),
                        int(enfd_val) + int(clsd_val),
                    ]
                ]
            data_row_array = np.array(data_row_list, int)
            del data_row_list
            # Check row totals
            assert_array_equal(row_total, np.einsum("ij->j", data_row_array[:, 2:4]))

            if row_key == TTL_KEY:
                col_totals = data_row_array
            else:
                invdata_array = ArrayBIGINT(
                    np.vstack((invdata_array, data_row_array))
                    if invdata_array.any()
                    else data_row_array
                )
            del data_row_array
        else:
            continue

    # Check column totals
    for _col_tot in col_totals:
        assert_array_equal(
            _col_tot[2:],
            np.einsum(
                "ij->j", invdata_array[invdata_array[:, 1] == _col_tot[1]][:, 2:]
            ),
        )

    return ArrayBIGINT(
        invdata_array[
            np.argsort(np.einsum("ij,ij->i", [[100, 1]], invdata_array[:, :2]))
        ]
    )


def _process_table_blks_cnt_type(
    _table_blocks: Sequence[Sequence[str]], /
) -> ArrayBIGINT:
    cnt_row_pat = re.compile(r"(\d+ (?:to \d+|\+)|TOTAL)")

    invdata_array: ArrayBIGINT = ArrayBIGINT(np.array([], int))
    col_totals = np.zeros(3, int)  # "enforced", "closed", "total"

    for _tbl_blk in _table_blocks:
        if cnt_row_pat.match(_blk_str := _tbl_blk[-3]):
            row_list_s = _blk_str.strip().replace(",", "").split("\n")
            row_list = np.array(
                [FCOUNT_HEADER_DICT[row_list_s[0]], *row_list_s[1:]], int
            )
            del row_list_s
            if row_list[3] != row_list[1] + row_list[2]:
                raise ValueError(
                    "Total number of investigations does not equal #enforced plus #closed."
                )
            if row_list[0] == TTL_KEY:
                col_totals = row_list
            else:
                invdata_array = ArrayBIGINT(
                    np.vstack((invdata_array, row_list))
                    if invdata_array.any()
                    else row_list
                )
        else:
            continue

    if not np.array_equal(
        np.array(list(col_totals[1:]), int), np.einsum("ij->j", invdata_array[:, 1:])
    ):
        raise ValueError("Column totals don't compute.")

    # mypy doesn't recognize that taking from an ArrayBIGINT returns an ArrayBIGINT
    # se we convert the result to shut it up; luckily, this appears to be costless
    return ArrayBIGINT(invdata_array[np.argsort(invdata_array[:, 0])])


def _download_invdata(_dl_path: Path = FID_WORK_DIR) -> tuple[str, ...]:
    if not _dl_path.is_dir():
        _dl_path.mkdir(parents=True)

    invdata_homepage_urls = (
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2005-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2007-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2011",
    )
    invdata_docnames = (
        "040831horizmergersdata96-03.pdf",
        "p035603horizmergerinvestigationdata1996-2005.pdf",
        "081201hsrmergerdata.pdf",
        "130104horizontalmergerreport.pdf",
    )

    if all(
        _dl_path.joinpath(invdata_docname).is_file()
        for invdata_docname in invdata_docnames
    ):
        return invdata_docnames

    invdata_docnames_dl: tuple[str, ...] = ()
    u3pm = urllib3.PoolManager()
    chunk_size_ = 1024 * 1024
    for invdata_homepage_url in invdata_homepage_urls:
        with u3pm.request(
            "GET", invdata_homepage_url, preload_content=False
        ) as _u3handle:
            invdata_soup = BeautifulSoup(_u3handle.data, "html.parser")
            invdata_attrs = [
                (_g.get("title", ""), _g.get("href", ""))
                for _g in invdata_soup.find_all("a")
                if _g.get("title", "") and _g.get("href", "").endswith(".pdf")
            ]
        for invdata_attr in invdata_attrs:
            invdata_docname, invdata_link = invdata_attr
            invdata_docnames_dl += (invdata_docname,)
            with (
                u3pm.request(
                    "GET", f"https://www.ftc.gov/{invdata_link}", preload_content=False
                ) as _urlopen_handle,
                _dl_path.joinpath(invdata_docname).open("wb") as invdata_fh,
            ):
                while True:
                    data = _urlopen_handle.read(chunk_size_)
                    if not data:
                        break
                    invdata_fh.write(data)

    return invdata_docnames_dl
