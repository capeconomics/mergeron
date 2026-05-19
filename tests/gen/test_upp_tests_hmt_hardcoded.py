"""Tests for the gen.upp_tests module."""

import numpy as np
import pytest

from mergeron import EMPTY_ARRAYBIGINT
from mergeron import ArrayBIGINT
from mergeron import RECForm
from mergeron import UPPAggregator
from mergeron.core.guidelines_boundaries import GuidelinesStandards
from mergeron.core.pseudorandom_numbers import seed_sequencer
from mergeron.gen import HMTSpec
from mergeron.gen import HSRFilingTest
from mergeron.gen import INVResolution
from mergeron.gen import PCMDistribution
from mergeron.gen import PCMRestriction
from mergeron.gen import PCMSpec
from mergeron.gen import PriceSpec
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution
from mergeron.gen import StatsGroup
from mergeron.gen import UPPTestRegime
from mergeron.gen.data_generation import MarketSample
from mergeron.gen.enforcement_stats import compute_enforcement_counts

SAMPLE_SIZE = 10**6
ENFT_THRESHOLDS = GuidelinesStandards(2023).presumption
ENFT_REGIME = UPPTestRegime(INVResolution.ENFT, UPPAggregator.AVG, UPPAggregator.AVG)

bench_set = (
    (
        (0.00, PriceSpec.PRICE_RND),
        {
            "ByFirmCount": ArrayBIGINT([
                [2, 325127, 310640, 317640, 325006, 317069],
                [3, 516619, 115152, 218098, 256993, 152189],
                [4, 100235, 14614, 38500, 27781, 18753],
                [5, 32214, 3133, 10950, 6260, 4096],
                [6, 14349, 849, 3961, 1897, 1157],
                [7, 6755, 280, 1646, 684, 389],
                [8, 3053, 109, 673, 274, 154],
                [9, 1313, 43, 263, 99, 53],
                [10, 335, 6, 52, 17, 11],
            ]),
            "ByDelta": ArrayBIGINT([
                [2500, 274890, 265859, 272252, 274876, 272960],
                [1200, 148071, 89558, 137195, 134447, 113007],
                [800, 97425, 32934, 68471, 66651, 42963],
                [500, 106940, 22860, 51946, 55171, 28489],
                [300, 100610, 14072, 28340, 37336, 16063],
                [200, 64381, 6671, 11975, 17964, 7182],
                [100, 81434, 6635, 11207, 17204, 6887],
                [0, 126249, 6237, 10397, 15362, 6320],
            ]),
            "ByHHIandDelta": EMPTY_ARRAYBIGINT,
        },
    ),
    (
        (0.01, PriceSpec.PRICE_RND),
        {
            "ByFirmCount": ArrayBIGINT([
                [2, 321370, 307884, 314730, 321317, 314759],
                [3, 509302, 114646, 218520, 265503, 153649],
                [4, 104075, 14391, 39062, 29099, 18836],
                [5, 34471, 3165, 11623, 6702, 4274],
                [6, 16199, 880, 4359, 2125, 1222],
                [7, 7977, 307, 1912, 807, 434],
                [8, 4037, 140, 889, 356, 190],
                [9, 1923, 55, 371, 135, 75],
                [10, 646, 12, 94, 35, 18],
            ]),
            "ByDelta": ArrayBIGINT([
                [2500, 279032, 269099, 275959, 279012, 276791],
                [1200, 158946, 93023, 144442, 143214, 118051],
                [800, 107499, 34334, 71866, 71990, 44938],
                [500, 119068, 23092, 53514, 58672, 28927],
                [300, 112395, 12864, 27620, 38496, 14936],
                [200, 72744, 5481, 10761, 17875, 5991],
                [100, 84884, 3337, 6957, 13470, 3550],
                [0, 65432, 250, 441, 3350, 273],
            ]),
            "ByHHIandDelta": EMPTY_ARRAYBIGINT,
        },
    ),
)


@pytest.mark.parametrize("_spec, _expected", bench_set)
def test_upp_tests_counts(_spec: dict, _expected: dict) -> None:
    """Test enforcement counts with sample restricted by HMT."""
    _lower_bound, _price_spec = _spec

    _market_sample = MarketSample(
        share_spec=ShareSpec(
            SHRDistribution.DIR_FLAT,
            lower_bound=_lower_bound,
            firm_counts_weights=np.ones(9),
            recapture_form=RECForm.INOUT,
            recapture_rate=ENFT_THRESHOLDS.rec,
        ),
        pcm_spec=PCMSpec(PCMDistribution.EMPR_M, pcm_restriction=PCMRestriction.MNL),
        price_spec=_price_spec,
        hsr_filing_test_type=HSRFilingTest.NONE,
        hmt_flag=HMTSpec(0, 0.05, 0.5, True),
        sample_size=SAMPLE_SIZE,
        seed_data=seed_sequencer(5),
        nthreads=8,
    )
    _market_sample.generate_sample()
    _market_sample.test_enforcement(ENFT_THRESHOLDS, ENFT_REGIME)

    _market_shares = _market_sample.dataset.shares
    _fcounts = np.einsum("ij->i", _market_shares > 0, dtype="<u8")[:, None]
    _fcounts_vals, _fcounts_counts = np.unique(_fcounts, return_counts=True)

    # Test _fcounts against firm-count values
    if not (
        (
            _fcounts_vals
            == 2 + np.arange(len(_market_sample.share_spec.firm_counts_weights))
        ).all()
    ):
        raise AssertionError(
            "DATA GENERATION ERROR: {} {} {}".format(
                "Generation of sample shares is inconsistent:",
                "In each draw, non-zero shares must number the firm counts implied by",
                "the specified firm-count weights.",
            )
        )

    # Test fcounts against firm-count weights
    if (
        _market_sample.pcm_spec.pcm_restriction != PCMRestriction.MNL
        or _market_sample.price_spec == PriceSpec.COST_SYM
    ) and not np.allclose(
        _fcw_test := _fcounts_counts / _fcounts.size,
        _market_sample.share_spec.firm_counts_weights,
        atol=1 / np.sqrt(SAMPLE_SIZE),
    ):
        print(_fcw_test, _market_sample.share_spec.firm_counts_weights, sep="\n")
        raise AssertionError(
            "DATA GENERATION ERROR: {} {} {}".format(
                "Generation of sample shares is inconsistent:",
                "array of drawn shares must exhibit firm counts",
                "in proportion to firm-count weights.",
            )
        )

    if _market_sample.enforcement_counts is not None and not (
        np.array_equal(
            _market_sample.enforcement_counts.ByFirmCount, _expected["ByFirmCount"]
        )
        and np.array_equal(
            _market_sample.enforcement_counts.ByDelta[::-1], _expected["ByDelta"]
        )
        and np.array_equal(
            compute_enforcement_counts(
                _market_sample.enforcement_counts.ByHHIandDelta[:, 1:], StatsGroup.DL
            )[::-1],
            _expected["ByDelta"],
        )
    ):
        print(
            "ByFirmCount:",
            repr(_market_sample.enforcement_counts.ByFirmCount),
            _expected["ByFirmCount"],
            sep="\n",
        )
        print(
            "ByDelta:",
            repr(_market_sample.enforcement_counts.ByDelta[::-1]),
            _expected["ByDelta"],
            sep="\n",
        )

        raise AssertionError("Enforcement counts differ")
