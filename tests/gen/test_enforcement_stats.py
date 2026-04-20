import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mergeron.core.ftc_merger_investigations_data as fid
from mergeron import ArrayBIGINT
from mergeron.core import INVDATA_ARCHIVE_PATH
from mergeron.gen import INVResolution
from mergeron.gen.enforcement_stats import IndustryGroup
from mergeron.gen.enforcement_stats import OtherEvidence
from mergeron.gen.enforcement_stats import StatsGroup
from mergeron.gen.enforcement_stats import enforcement_counts_observed_by_tabletype

invdata_array_dict = fid.construct_data(
    INVDATA_ARCHIVE_PATH,
    flag_backward_compatibility=False,
    flag_pharma_for_exclusion=True,
)


@pytest.mark.parametrize(
    "_stats_group, _test_val",
    zip(
        (StatsGroup.FC, StatsGroup.DL), np.array([[573, 132], [780, 173]]), strict=True
    ),
)
def test_enf_stats(_stats_group: StatsGroup, _test_val: ArrayBIGINT) -> None:
    """Test enforcement counts by group."""
    enf_spec_ = INVResolution.CLRN
    enf_cnts_ = enforcement_counts_observed_by_tabletype(
        invdata_array_dict,
        "1996-2003",
        IndustryGroup.ALL,
        OtherEvidence.UNR,
        _stats_group,
        enf_spec_,
    )[:, -2:]
    enf_cnts_totals = np.einsum("ij->j", enf_cnts_)
    assert_array_equal(enf_cnts_totals, _test_val)
