from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mergeron.core.ftc_merger_investigations_data as fid
from mergeron.core import INVData

invdata = fid.construct_data(fid.INVDATA_ARCHIVE_PATH)

test_dict = {
    "1996-2003": {
        "ByHHIandDelta": {
            "Table 3.1": (607, 173),
            "Table 3.2": (129, 23),
            "Table 3.3": (208, 68),
            "Table 3.4": (58, 7),
            "Table 3.5": (24, 1),
            "Table 3.6": (188, 74),
            "Table 5.1": (18, 2),
            "Table 5.2": (71, 37),
            "Table 7.1": (50, 1),
            "Table 7.2": (31, 34),
            "Table 9.1": (0, 19),
            "Table 9.2": (89, 20),
            "Table 9.X": (518, 134),
            "Table 3.X": (395, 98),
        },
        "ByFirmCount": {
            "Table 4.1": (441, 132),
            "Table 4.2": (129, 23),
            "Table 4.3": (53, 25),
            "Table 4.4": (58, 7),
            "Table 4.5": (24, 1),
            "Table 4.6": (177, 76),
            "Table 6.1": (18, 2),
            "Table 6.2": (71, 37),
            "Table 8.1": (50, 1),
            "Table 8.2": (31, 34),
            "Table 10.1": (0, 19),
            "Table 10.2": (89, 20),
            "Table 10.X": (352, 93),
            "Table 4.X": (240, 55),
        },
    },
    "1996-2005": {
        "ByHHIandDelta": {
            "Table 3.1": (744, 228),
            "Table 3.2": (8, 5),
            "Table 3.3": (113, 33),
            "Table 3.4": (19, 9),
            "Table 3.5": (17, 11),
            "Table 3.6": (124, 55),
            "Table 5.1": (22, 3),
            "Table 5.2": (95, 54),
            "Table 7.1": (72, 1),
            "Table 7.2": (36, 49),
            "Table 9.1": (0, 30),
            "Table 9.2": (117, 27),
            "Table 9.X": (627, 171),
        },
        "ByFirmCount": {
            "Table 4.1": (578, 169),
            "Table 4.2": (8, 5),
            "Table 4.3": (30, 11),
            "Table 4.4": (19, 9),
            "Table 4.5": (17, 11),
            "Table 4.6": (124, 38),
            "Table 6.1": (22, 3),
            "Table 6.2": (95, 54),
            "Table 8.1": (72, 1),
            "Table 8.2": (36, 49),
            "Table 10.1": (0, 30),
            "Table 10.2": (117, 27),
            "Table 10.X": (461, 112),
        },
    },
    "1996-2007": {
        "ByHHIandDelta": {
            "Table 3.1": (870, 280),
            "Table 3.2": (134, 19),
            "Table 3.3": (213, 66),
            "Table 3.4": (64, 10),
            "Table 3.5": (53, 12),
            "Table 3.6": (282, 134),
            "Table 5.1": (22, 3),
            "Table 5.2": (109, 64),
            "Table 7.1": (83, 2),
            "Table 7.2": (37, 55),
            "Table 9.1": (0, 36),
            "Table 9.2": (131, 31),
            "Table 9.X": (739, 213),
        },
        "ByFirmCount": {
            "Table 4.1": (704, 221),
            "Table 4.2": (134, 19),
            "Table 4.3": (58, 23),
            "Table 4.4": (64, 10),
            "Table 4.5": (53, 12),
            "Table 4.6": (282, 119),
            "Table 6.1": (22, 3),
            "Table 6.2": (109, 64),
            "Table 8.1": (83, 2),
            "Table 8.2": (37, 55),
            "Table 10.1": (0, 36),
            "Table 10.2": (131, 31),
            "Table 10.X": (573, 154),
        },
    },
    "1996-2011": {
        "ByHHIandDelta": {
            "Table 3.1": (1055, 304),
            "Table 3.2": (152, 24),
            "Table 3.3": (214, 69),
            "Table 3.4": (90, 13),
            "Table 3.5": (119, 3),
            "Table 3.6": (8, 12),
            "Table 3.7": (15, 10),
            "Table 3.8": (21, 19),
            "Table 3.9": (436, 154),
            "Table 5.1": (25, 3),
            "Table 5.2": (150, 80),
            "Table 7.1": (111, 3),
            "Table 7.2": (53, 69),
            "Table 9.1": (0, 45),
            "Table 9.2": (175, 38),
            "Table 9.X": (880, 221),
            "Table 3.X": (456, 106),
        },
        "ByFirmCount": {
            "Table 4.1": (898, 245),
            "Table 4.2": (152, 24),
            "Table 4.3": (59, 27),
            "Table 4.4": (90, 13),
            "Table 4.5": (119, 3),
            "Table 4.6": (8, 12),
            "Table 4.7": (15, 11),
            "Table 4.8": (21, 19),
            "Table 4.9": (434, 136),
            "Table 6.1": (25, 3),
            "Table 6.2": (150, 80),
            "Table 8.1": (111, 3),
            "Table 8.2": (53, 69),
            "Table 10.1": (0, 45),
            "Table 10.2": (175, 38),
            "Table 10.X": (723, 162),
            "Table 4.X": (301, 64),
        },
    },
    "2004-2011": {
        "ByHHIandDelta": {
            "Table 3.1": (448, 131),
            "Table 3.2": (23, 1),
            "Table 3.3": (6, 3),
            "Table 3.4": (32, 6),
            "Table 3.5": (95, 13),
            "Table 5.1": (7, 1),
            "Table 5.2": (80, 44),
            "Table 7.1": (61, 2),
            "Table 7.2": (23, 36),
            "Table 9.1": (0, 26),
            "Table 9.2": (87, 18),
            "Table 9.X": (361, 87),
            "Table 3.X": (61, 10),
        },
        "ByFirmCount": {
            "Table 4.1": (457, 113),
            "Table 4.2": (23, 1),
            "Table 4.3": (6, 2),
            "Table 4.4": (32, 6),
            "Table 4.5": (95, 13),
            "Table 6.1": (7, 1),
            "Table 6.2": (79, 43),
            "Table 8.1": (61, 2),
            "Table 8.2": (22, 35),
            "Table 10.1": (0, 27),
            "Table 10.2": (86, 18),
            "Table 4.X": (61, 9),
        },
    },
}


def unnest_dict_to_list(_dict: Mapping[str, Any]) -> list[str | tuple[int, int]]:
    retval = []
    for _k, _v in _dict.items():
        retval += (
            [[_k, *_l] for _l in unnest_dict_to_list(_v)]
            if isinstance(_v, dict)
            else [[_k, _v]]
        )
    return retval


@pytest.mark.parametrize(
    "_data_period, _table_type, _table_no, _test_val", unnest_dict_to_list(test_dict)
)
def test_invdata(
    _data_period: str,
    _table_type: str,
    _table_no: str,
    _test_val: tuple[int, int],
    _fid_data: INVData = invdata,
) -> None:
    invdata_tots = np.einsum(
        "ij->j", _fid_data[_data_period][_table_type][_table_no].data_array[:, -3:]
    )
    invdata_test = np.array([*_test_val, np.sum(_test_val)], int)

    assert_array_equal(invdata_tots, invdata_test)
