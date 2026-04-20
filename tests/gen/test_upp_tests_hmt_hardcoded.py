"""Tests for the gen.upp_tests module."""

import sys

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
from mergeron.gen import UPPTestRegime
from mergeron.gen.data_generation import MarketSample

SAMPLE_SIZE = 10**6
ENFT_THRESHOLDS = GuidelinesStandards(2023).presumption
ENFT_REGIME = UPPTestRegime(INVResolution.ENFT, UPPAggregator.AVG, UPPAggregator.AVG)

bench_set = (
    (
        (0.00, PriceSpec.PRICE_RND),
        {
            "ByFirmCount": ArrayBIGINT([
                [2, 325127, 322781, 323967, 298317, 297993],
                [3, 516619, 150896, 262333, 298302, 198190],
                [4, 100235, 18501, 46869, 34087, 24930],
                [5, 32214, 4052, 13766, 7952, 5689],
                [6, 14349, 1122, 5246, 2504, 1681],
                [7, 6755, 393, 2297, 916, 612],
                [8, 3053, 156, 968, 362, 232],
                [9, 1313, 57, 393, 132, 87],
                [10, 335, 11, 88, 25, 16],
            ]),
            "ByDelta": ArrayBIGINT([
                [2500, 274890, 273488, 274714, 250246, 250221],
                [1200, 148071, 108259, 144854, 140856, 129439],
                [800, 97425, 43219, 81202, 76408, 57495],
                [500, 106940, 30520, 67801, 65578, 40983],
                [300, 100610, 18266, 41531, 45928, 23402],
                [200, 64381, 8412, 16817, 22447, 10123],
                [100, 81434, 8159, 15128, 21645, 9297],
                [0, 126249, 7646, 13880, 19489, 8470],
            ]),
            "ByHHIandDelta": EMPTY_ARRAYBIGINT,
        },
    ),
    (
        (0.01, PriceSpec.PRICE_RND),
        {
            "ByFirmCount": ArrayBIGINT([
                [2, 321370, 319650, 320615, 293775, 293688],
                [3, 509302, 151859, 264772, 307529, 201927],
                [4, 104075, 18450, 48197, 35816, 25315],
                [5, 34471, 4163, 14809, 8512, 5957],
                [6, 16199, 1185, 5883, 2831, 1768],
                [7, 7977, 423, 2720, 1081, 663],
                [8, 4037, 199, 1293, 473, 294],
                [9, 1923, 77, 586, 188, 109],
                [10, 646, 20, 158, 48, 30],
            ]),
            "ByDelta": ArrayBIGINT([
                [2500, 279032, 277491, 278846, 253333, 253301],
                [1200, 158946, 113414, 154207, 150714, 136935],
                [800, 107499, 45393, 86512, 83046, 60752],
                [500, 119068, 30831, 70528, 70401, 41739],
                [300, 112395, 16910, 41733, 47962, 22257],
                [200, 72744, 7126, 15854, 22770, 8800],
                [100, 84884, 4451, 10388, 17606, 5470],
                [0, 65432, 410, 965, 4421, 497],
            ]),
            "ByHHIandDelta": EMPTY_ARRAYBIGINT,
        }
        if sys.platform == "linux"
        else {
            "ByFirmCount": ArrayBIGINT([
                [2, 321370, 319650, 320615, 293776, 293689],
                [3, 509302, 151859, 264772, 307529, 201927],
                [4, 104075, 18450, 48197, 35816, 25315],
                [5, 34471, 4163, 14809, 8512, 5957],
                [6, 16199, 1185, 5883, 2831, 1768],
                [7, 7977, 423, 2720, 1081, 663],
                [8, 4037, 199, 1293, 473, 294],
                [9, 1923, 77, 586, 188, 109],
                [10, 646, 20, 158, 48, 30],
            ]),
            "ByDelta": ArrayBIGINT([
                [2500, 279032, 277491, 278846, 253334, 253302],
                [1200, 158946, 113414, 154207, 150714, 136935],
                [800, 107499, 45393, 86512, 83046, 60752],
                [500, 119068, 30831, 70528, 70401, 41739],
                [300, 112395, 16910, 41733, 47962, 22257],
                [200, 72744, 7126, 15854, 22770, 8800],
                [100, 84884, 4451, 10388, 17606, 5470],
                [0, 65432, 410, 965, 4421, 497],
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
    _market_sample.compute_enforcement_counts(ENFT_THRESHOLDS, ENFT_REGIME)

    _share_array = _market_sample.dataset.share_array
    _fcounts = np.einsum("ij->i", _share_array > 0, dtype="<u8")[:, None]
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

    if not (
        np.array_equal(
            _market_sample.enforcement_counts.ByFirmCount, _expected["ByFirmCount"]
        )
        and np.array_equal(
            _market_sample.enforcement_counts.ByDelta[::-1], _expected["ByDelta"]
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
