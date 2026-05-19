"""Tests for the gen.upp_tests module."""

from itertools import product as iter_prod
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mergeron import YAML
from mergeron import RECForm
from mergeron import UPPAggregator
from mergeron import zipfile
from mergeron.core.guidelines_boundaries import GuidelinesStandards
from mergeron.core.pseudorandom_numbers import prng
from mergeron.core.pseudorandom_numbers import seed_sequencer
from mergeron.gen import HMTSpec
from mergeron.gen import HSRFilingTest
from mergeron.gen import INVResolution
from mergeron.gen import PCMDistribution
from mergeron.gen import PCMRestriction
from mergeron.gen import PCMSpec
from mergeron.gen import PriceSpec
from mergeron.gen import SeedSequenceData
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution
from mergeron.gen import UPPTestRegime
from mergeron.gen.data_generation import MarketSample
from mergeron.gen.data_generation_functions import compute_merging_firm_diversion_ratios

ZipFile = zipfile.ZipFile

TEST_DATA_DIR = Path(__file__).parents[1] / "data"
ENFT_THRESHOLDS = GuidelinesStandards(2023).presumption
ENFT_REGIME = UPPTestRegime(INVResolution.ENFT, UPPAggregator.AVG, UPPAggregator.AVG)


SAMPLE_SPEC_AUX = {"sample_size": 10**4, "nthreads": 8}

specs_path = (Path(__file__).parents[1] / "data").joinpath(
    "test_upp_tests_hmt_archived_specs_concentrating_firm_counts.zip"
)
with ZipFile(specs_path, "r") as _zh, _zh.open("concentrated_subset.yaml") as _zfh:
    firm_count_concentrating_subset = YAML.load(_zfh)

bench_set = tuple(
    iter_prod(
        (0.0, 0.01),
        PCMDistribution.__members__.keys(),
        PCMRestriction.__members__.keys(),
        PriceSpec.__members__.keys(),
        HSRFilingTest.__members__.keys(),
    )
)
bench_set_choice = tuple(
    bench_set[_i]
    for _i in prng().choice(range(len(bench_set)), size=280, replace=False)
)


@pytest.mark.parametrize("_spec", bench_set)
def test_upp_tests_counts(_spec: tuple[float, str, str, str, str]) -> None:
    """Test enforcement counts with sample restricted by HMT."""
    print(_spec)
    _archive_name = "upp_test_data_hmt_-{}-{}.zip".format(
        f"{float(_spec[0]) * 100:1.0f}PCT", "-".join(_spec[1:])
    )

    _lower_bound, _pcm_distribution, _pcm_restriction, _price_spec, _hsr_filing_test = (
        _spec
    )
    _lower_bound = float(_lower_bound)
    _pcm_distribution = PCMDistribution[_pcm_distribution]
    _pcm_restriction = PCMRestriction[_pcm_restriction]
    _price_spec = PriceSpec[_price_spec]
    _hsr_filing_test = HSRFilingTest[_hsr_filing_test]

    market_sample = MarketSample(
        share_spec=ShareSpec(
            SHRDistribution.DIR_FLAT,
            firm_counts_weights=np.ones(9),
            lower_bound=_lower_bound,
            recapture_form=RECForm.INOUT,
            recapture_rate=ENFT_THRESHOLDS.rec,
        ),
        pcm_spec=PCMSpec(_pcm_distribution, pcm_restriction=_pcm_restriction),
        price_spec=_price_spec,
        hsr_filing_test_type=_hsr_filing_test,
        seed_data=seed_sequencer(len(SeedSequenceData.__attrs_attrs__)),
        hmt_flag=HMTSpec(0, 0.05, 0.5, True),
        **SAMPLE_SPEC_AUX,
    )
    market_sample.generate_sample()
    if market_sample.dataset is None:
        raise AssertionError("No dataset generated.")

    _market_shares = market_sample.dataset.shares
    _aggr_purch_prob = market_sample.dataset.aggregate_choice_probability

    # Test market shares
    if len(_market_shares) != SAMPLE_SPEC_AUX["sample_size"]:
        for _s in _spec:
            print(_s)
        print(len(_market_shares), "=?", SAMPLE_SPEC_AUX["sample_size"])
        raise AssertionError(
            "Generated share array does not match specified sample size in length."
        )
    try:
        assert_allclose(
            np.einsum("ij->", _market_shares), float(SAMPLE_SPEC_AUX["sample_size"])
        )
    except AssertionError as _err:
        print(np.einsum("ij->", _market_shares), "=?", SAMPLE_SPEC_AUX["sample_size"])
        raise _err

    # Test diversion ratios, as ordinals
    _mrgng_firm_shares = _market_shares[:, :2]
    _diversion_array = compute_merging_firm_diversion_ratios(
        market_sample.share_spec.recapture_form,
        market_sample.share_spec.recapture_rate,
        _mrgng_firm_shares,
        _aggr_purch_prob,
    )
    divr_assert_test = (
        (np.round(np.einsum("ij->i", _mrgng_firm_shares), 15) == 1)
        | (np.argmin(_mrgng_firm_shares, axis=1) == np.argmax(_diversion_array, axis=1))
    )[:, None]
    if not all(divr_assert_test):
        print(_mrgng_firm_shares, _diversion_array)
        raise ValueError(
            "{} {} {} {}".format(
                "Data construction fails tests:",
                "the index of min(s_1, s_2) must equal",
                "the index of max(d_12, d_21), for all draws.",
                "unless frmshr_array sums to 1.00.",
            )
        )
    del _mrgng_firm_shares, _diversion_array, divr_assert_test

    # Test aggregate purchase probability
    _choice_probabilities = np.einsum("ij,ij->ij", _aggr_purch_prob, _market_shares)
    assert_allclose(
        np.einsum("ij->i", _choice_probabilities)[:, None], _aggr_purch_prob
    )

    # Test diversion ratios, as cardinals
    _prod_idx = 0
    divratio_1j = np.divide(
        _choice_probabilities, 1 - _choice_probabilities[:, [_prod_idx]]
    )
    divratio_1j[:, _prod_idx] = 0

    assert_allclose(
        np.einsum("ij->i", divratio_1j)[:, None]
        + np.divide(1 - _aggr_purch_prob, 1 - _choice_probabilities[:, [_prod_idx]]),
        np.ones_like(_aggr_purch_prob),
    )
    del _choice_probabilities, _prod_idx, divratio_1j

    # Test firm counts
    if _spec not in firm_count_concentrating_subset:
        _fcounts = np.einsum("ij->i", _market_shares > 0, dtype="<u1")[:, None]
        _fcounts_vals = np.unique(_fcounts)
        # Test firm counts against expected firm-count values
        if not np.array_equal(
            _fcounts_vals,
            _fk := 2 + np.arange(len(market_sample.share_spec.firm_counts_weights)),
        ):
            print("Firm counts:", _fcounts_vals, _fk, sep="\n")
            raise AssertionError("Firm count calculation failed.")
        del _fcounts, _fcounts_vals
    del _market_shares, _aggr_purch_prob

    market_sample.test_enforcement(ENFT_THRESHOLDS, ENFT_REGIME)

    if (
        market_sample.enforcement_counts.ByFirmCount[:, 1].sum()
        != SAMPLE_SPEC_AUX["sample_size"]
    ):
        raise AssertionError(
            "Generated sample size does not match benchmark specification."
        )

    object.__setattr__(market_sample, "dataset", None)
    # Store the data at the time the tests are created
    if not (_zap := TEST_DATA_DIR / _archive_name).is_file():
        with ZipFile(_zap, "w", 93) as _zaf:
            _zpath = zipfile.Path(_zaf, at="")
            with (_zpath / "mergeron_market_sample_enf_counts.yaml").open("w") as _yfh:
                YAML.dump(market_sample.enforcement_counts, _yfh)

    # Load benchmark data for the test
    with ZipFile(_zap, "r") as _zaf:
        _zpath = zipfile.Path(_zaf, at="")
        with (_zpath / "mergeron_market_sample_enf_counts.yaml").open("r") as _yfh:
            _upp_test_counts_bechmark = YAML.load(_yfh)

    if market_sample.enforcement_counts and not all((
        np.array_equal(
            market_sample.enforcement_counts.ByFirmCount,
            _upp_test_counts_bechmark.ByFirmCount,
        ),
        np.array_equal(
            market_sample.enforcement_counts.ByDelta, _upp_test_counts_bechmark.ByDelta
        ),
        np.array_equal(
            market_sample.enforcement_counts.ByHHIandDelta,
            _upp_test_counts_bechmark.ByHHIandDelta,
        ),
    )):
        for _s in _spec:
            print(_s)
        print("Test values:")
        print("ByFirmCount:", repr(market_sample.enforcement_counts.ByFirmCount))
        print("ByDelta:", repr(market_sample.enforcement_counts.ByDelta[::-1]))
        print("ByHHIandDelta:", repr(market_sample.enforcement_counts.ByHHIandDelta))

        print()
        print("Benchmark values:")
        print("ByFirmCount:", repr(_upp_test_counts_bechmark.ByFirmCount))
        print("ByDelta:", repr(_upp_test_counts_bechmark.ByDelta[::-1]))
        print("ByHHIandDelta:", repr(_upp_test_counts_bechmark.ByHHIandDelta))

        raise AssertionError("Enforcement counts differ")
