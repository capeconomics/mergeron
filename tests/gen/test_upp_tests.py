"""Tests for the gen.upp_tests module."""

import gc

import numpy as np
import pendulum
import pytest

import mergeron.core.guidelines_boundaries as gbl
import mergeron.gen.enforcement_stats as esl
from mergeron import ArrayBIGINT
from mergeron import RECForm
from mergeron import UPPAggregator
from mergeron.core.pseudorandom_numbers import seed_sequencer
from mergeron.gen import INVResolution
from mergeron.gen import SeedSequenceData
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution
from mergeron.gen import UPPTestRegime
from mergeron.gen.data_generation import MarketSample

SAMPLE_SIZE = 10**6

BENCH_VALS_DICT = {
    esl.StatsGroup.FC: ArrayBIGINT([
        [2, 235125, 2815, 0, 270, 359],
        [3, 326274, 29093, 13844, 10069, 20087],
        [4, 237921, 40655, 25717, 14848, 31747],
        [5, 92442, 22854, 16772, 8652, 18961],
        [6, 56721, 18164, 14766, 6989, 15596],
        [7, 17783, 7013, 6080, 2766, 6197],
        [8, 21267, 9713, 8787, 3722, 8593],
        [9, 7159, 3766, 3495, 1497, 3388],
        [10, 5308, 3015, 2858, 1203, 2769],
    ]),
    esl.StatsGroup.DL: ArrayBIGINT([
        [0, 72709, 52185, 50716, 21824, 49289],
        [100, 69309, 27530, 22387, 9563, 22916],
        [200, 59369, 15760, 10242, 5259, 12129],
        [300, 92799, 16490, 7090, 5548, 11330],
        [500, 104883, 10689, 1884, 3714, 6392],
        [800, 104233, 5965, 0, 2055, 3086],
        [1200, 216168, 5917, 0, 1720, 2181],
        [2500, 280530, 2552, 0, 333, 374],
    ]),
    esl.StatsGroup.HD: ArrayBIGINT([
        [0, 0, 3850, 2977, 2957, 1261, 2859],
        [0, 100, 2256, 1187, 1170, 367, 1037],
        [0, 200, 937, 337, 329, 108, 260],
        [0, 300, 426, 110, 102, 50, 97],
        [0, 500, 10, 4, 4, 1, 3],
        [0, 800, 0, 0, 0, 0, 0],
        [0, 1200, 0, 0, 0, 0, 0],
        [0, 2500, 0, 0, 0, 0, 0],
        [1800, 0, 3055, 2322, 2289, 935, 2172],
        [1800, 100, 2119, 1055, 991, 362, 892],
        [1800, 200, 1205, 418, 374, 132, 336],
        [1800, 300, 1064, 251, 207, 98, 188],
        [1800, 500, 119, 17, 14, 4, 9],
        [1800, 800, 0, 0, 0, 0, 0],
        [1800, 1200, 0, 0, 0, 0, 0],
        [1800, 2500, 0, 0, 0, 0, 0],
        [2000, 0, 6804, 5038, 4944, 2086, 4725],
        [2000, 100, 5203, 2424, 2188, 838, 2048],
        [2000, 200, 3649, 1208, 978, 430, 1004],
        [2000, 300, 4269, 955, 647, 348, 722],
        [2000, 500, 1758, 220, 110, 98, 153],
        [2000, 800, 45, 1, 0, 0, 0],
        [2000, 1200, 0, 0, 0, 0, 0],
        [2000, 2500, 0, 0, 0, 0, 0],
        [2400, 0, 10807, 7576, 7350, 3157, 7146],
        [2400, 100, 9716, 4110, 3443, 1446, 3489],
        [2400, 200, 7418, 2330, 1652, 790, 1796],
        [2400, 300, 10682, 2173, 1214, 795, 1604],
        [2400, 500, 8712, 1115, 356, 435, 741],
        [2400, 800, 2472, 179, 0, 77, 105],
        [2400, 1200, 10, 0, 0, 0, 0],
        [2400, 2500, 0, 0, 0, 0, 0],
        [3000, 0, 15503, 10254, 9771, 4341, 9670],
        [3000, 100, 18428, 6940, 5358, 2551, 5912],
        [3000, 200, 15349, 4414, 2847, 1503, 3463],
        [3000, 300, 24680, 4915, 2108, 1725, 3479],
        [3000, 500, 27443, 3127, 621, 1190, 2088],
        [3000, 800, 23389, 1637, 0, 634, 948],
        [3000, 1200, 7596, 383, 0, 143, 197],
        [3000, 2500, 0, 0, 0, 0, 0],
        [4000, 0, 11921, 8915, 8701, 3650, 8355],
        [4000, 100, 10508, 3995, 3133, 1330, 3188],
        [4000, 200, 8767, 2233, 1310, 739, 1757],
        [4000, 300, 14306, 2589, 928, 838, 1732],
        [4000, 500, 17797, 1929, 289, 705, 1204],
        [4000, 800, 20631, 1338, 0, 445, 699],
        [4000, 1200, 34776, 1314, 0, 496, 635],
        [4000, 2500, 0, 0, 0, 0, 0],
        [5000, 0, 11920, 7683, 7284, 3319, 7254],
        [5000, 100, 18349, 6219, 4532, 2201, 4977],
        [5000, 200, 17947, 4515, 2752, 1522, 3427],
        [5000, 300, 29188, 4961, 1884, 1631, 3386],
        [5000, 500, 35782, 3731, 490, 1211, 2090],
        [5000, 800, 39772, 2316, 0, 831, 1233],
        [5000, 1200, 105563, 3103, 0, 915, 1150],
        [5000, 2500, 33112, 549, 0, 120, 139],
        [7000, 0, 8849, 7420, 7420, 3075, 7108],
        [7000, 100, 2730, 1600, 1572, 468, 1373],
        [7000, 200, 4097, 305, 0, 35, 86],
        [7000, 300, 8184, 536, 0, 63, 122],
        [7000, 500, 13262, 546, 0, 70, 104],
        [7000, 800, 17924, 494, 0, 68, 101],
        [7000, 1200, 68223, 1117, 0, 166, 199],
        [7000, 2500, 247418, 2003, 0, 213, 235],
    ]),
}


@pytest.fixture(scope="module")
def test_market_sample() -> MarketSample:
    """Create test sample of UPP tests.

    Tests by firm-count, delta HHI, and post-merger HHI.
    """
    _pub_year = 2010
    _mg_thresholds0 = gbl.GuidelinesStandards(_pub_year).safeharbor
    _upp_test_regime0 = UPPTestRegime(
        INVResolution.CLRN, UPPAggregator.MAX, UPPAggregator.MAX
    )

    _firm_count_weights = np.array([133, 184, 134, 52, 32, 10, 12, 4, 3], int)

    market_sample = MarketSample(
        share_spec=ShareSpec(
            SHRDistribution.DIR_FLAT,
            firm_counts_weights=_firm_count_weights,
            parameters=np.ones(1 + len(_firm_count_weights), float),
            lower_bound=1e-2,
            recapture_form=RECForm.INOUT,
            recapture_rate=_mg_thresholds0.rec,
        ),
        hmt_flag=False,
        sample_size=SAMPLE_SIZE,
        seed_data=seed_sequencer(len(SeedSequenceData.__attrs_attrs__)),
        nthreads=16,
    )

    start_time = pendulum.now()
    market_sample.test_enforcement(_mg_thresholds0, _upp_test_regime0)

    total_duration = pendulum.now().diff(start_time).total_seconds()
    print(f"Estimations completed in total duration of {total_duration:.6f} secs.")

    if market_sample.enforcement_counts is None:
        raise ValueError("Enforcement counts estimation was unsuccessful.")

    yield market_sample
    del market_sample


@pytest.mark.parametrize("_stats_group, _bench_val", BENCH_VALS_DICT.items())
def test_enforcement_counts(
    _stats_group: esl.StatsGroup,
    _bench_val: ArrayBIGINT,
    test_market_sample: MarketSample,
) -> None:
    """Test enforcement counts by group."""
    _test_val = getattr(test_market_sample.enforcement_counts, _stats_group.value)

    if not np.array_equal(_test_val, _bench_val):
        print(f"Stats group: {_stats_group}")
        print(repr(_bench_val))
        print(repr(_test_val))
        raise AssertionError(f"Enforcement counts differ for {_stats_group}")


gc.collect()
