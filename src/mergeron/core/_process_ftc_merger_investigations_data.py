"""Download and parse FTC Merger Investigations Data."""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import urllib3
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .. import ArrayBIGINT
from .. import _mappingproxy_from_mapping
from . import DELTA_HEADER_DICT
from . import FCOUNT_HEADER_DICT
from . import FID_WORK_DIR
from . import HHI_HEADER_DICT
from . import TABLE_TYPES
from . import TOT_KEY
from . import INVData
from . import INVDataDict
from . import INVTableData

TABLE_NO_RE = re.compile(r"Table\s+(\d+)\s?\.(\d+)")

_FC_ROW_RE = re.compile(r"((?:\d{1,2} to \d)|(?:10 \+)|TOTAL) ([,\d]+ [,\d]+ [,\d]+)")
_HHI_ROW_RE = re.compile(
    r"((?:0 - 1,799)|(?:\d,\d{3} - \d,\d{3})|(?:7,000 \+)|TOTAL) (.*)"
)
_DATA_PERIOD_RE = re.compile(r"(\d{4}) *(-) *(\d{4})")


def _parse_invdata() -> INVData:
    """Parse FTC merger investigations data reports to structured data.

    Returns
    -------
    Immutable dictionary of merger investigations data, keyed to
    reporting period, and including all tables organized by
    Firm Count (number of remaining competitors) and
    by range of HHI and ∆HHI.
    """
    # Initialize containers for parsed data
    invdata: INVDataDict = {}

    invdata_docnames = _download_invdata(FID_WORK_DIR)
    for invdata_docname in invdata_docnames:
        invdata_pdf_path = FID_WORK_DIR.joinpath(invdata_docname)
        invdata_doc = PdfReader(invdata_pdf_path)
        invdata = _parse_tables(invdata, invdata_doc)
        invdata_doc.close()

    return _mappingproxy_from_mapping(invdata)


def _parse_tables(_invdata: INVDataDict, _invdata_doc: PdfReader) -> INVDataDict:
    _doc_title = (
        ", ".join(("Horizontal Merger Investigation Data", "Fiscal Years", "1996-2005"))
        if (_t := _invdata_doc.metadata["/Title"]) == " "  # type: ignore[index]
        else _t
    )

    data_period = "".join(_DATA_PERIOD_RE.findall(_doc_title)[0])  # type: ignore[arg-type]
    _invdata[data_period] = {}

    table_start, table_, table_no = False, [], ""
    for page_ in _invdata_doc.pages:
        for line_ in page_.extract_text().splitlines():
            if table_start:
                table_ += [line_]
                if line_.startswith("TOTAL"):
                    _invdata[data_period][table_no] = _parse_lines(
                        data_period, table_no, table_
                    )
                    table_start, table_, table_no = False, [], ""
            elif re.fullmatch(r"(Table\s+\d{1,2}\s*\.\d)", line_):
                table_start, table_, table_no = True, [line_], line_

    return _invdata


def _parse_lines(
    _data_period: str, _table_no: str, _table: Sequence[str]
) -> INVTableData:
    _table_type = _get_table_type(_table_no)
    _igroup = _table[2 if _data_period == "1996-2011" else 3].replace(
        "Pharmaceutical ", "Pharmaceuticals "
    )
    _oevid = (
        _table[3 if _data_period == "1996-2011" else 5]
        if int(TABLE_NO_RE.fullmatch(_table_no)[1]) > 4  # type: ignore[index]
        else "Unrestricted on additional evidence"
    )

    # Construct data array, test, and drop TOTAL rows
    if _table_type == TABLE_TYPES[1]:
        array_ = ArrayBIGINT([
            (
                FCOUNT_HEADER_DICT[_f],
                *[int(_h.replace(",", "")) for _h in _g.split(" ")],
            )
            for _row in _table[-11:]
            for _f, _g in [_FC_ROW_RE.fullmatch(_row).groups()]  # type: ignore[union-attr]
        ])

        # Test TOTAL
        if not np.array_equal(
            array_[array_[:, 0] != TOT_KEY][:, 1:].sum(axis=0),
            array_[array_[:, 0] == TOT_KEY][0, 1:],
        ):
            raise ValueError("Table totals do not match for pre-merger firm counts")

        # Drop TOTAL
        array_ = array_[array_[:, 0] != TOT_KEY]

    else:
        array_ = ArrayBIGINT([
            (
                HHI_HEADER_DICT[_h.replace("7,000 +", "7,000 - 10,000")],
                _d,
                *[int(_k.replace(",", "")) for _k in _ec.split("/")],
            )
            for _row in _table[-9:]
            for _h, _v in [_HHI_ROW_RE.fullmatch(_row).groups()]  # type: ignore[union-attr]
            for _d, _ec in zip(DELTA_HEADER_DICT.values(), _v.split(" "), strict=True)
        ])
        # Add column for total number of investigated mergers
        array_ = np.hstack((array_, array_[:, -2:].sum(axis=1, keepdims=True)))

        # Test TOTAL
        if not np.array_equal(
            array_[array_[:, 0] != TOT_KEY][:, 2:].sum(axis=0),
            array_[array_[:, 0] == TOT_KEY][:, 2:].sum(axis=0),
        ):
            raise ValueError("Table totals do not match for post-merger HHIs")

        if not all(
            np.array_equal(
                _a[_a[:, 1] != TOT_KEY][:, 2:].sum(axis=0),
                _a[_a[:, 1] == TOT_KEY][0, 2:],
            )
            for _h in np.unique(array_[:, 0])
            for _a in [array_[array_[:, 0] == _h]]
        ):
            raise ValueError(
                "Table totals do not match with pre-merger HHI values, across deltas"
            )

        # Drop TOTAL
        array_ = (_arr := array_[array_[:, 0] != TOT_KEY])[_arr[:, 1] != TOT_KEY]

    return INVTableData(_table_type, _igroup, _oevid, array_)


def _get_table_type(_table_no: str) -> str:
    return TABLE_TYPES[(int(TABLE_NO_RE.fullmatch(_table_no)[1]) + 1) % 2]  # type: ignore[index]


def print_tables(_reader: PdfReader) -> None:
    """Print tables in pages of FTC merger investigations data report.

    The report is downloaded from the FTC website and loaded with
    :mod:`pypdf.PdfReader`, with the result as input to this function.
    """
    table_start, table_, table_no = False, [], ""
    for page_ in _reader.pages:
        for line_ in page_.extract_text().splitlines():
            if table_start:
                table_ += [line_]
                if line_.startswith("TOTAL"):
                    print(table_no)
                    print("\n".join(table_))
                    print("\n\n")
                    table_start, table_, table_no = False, [], ""
            elif TABLE_NO_RE.fullmatch(line_):
                table_start, table_, table_no = True, [line_], line_


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
