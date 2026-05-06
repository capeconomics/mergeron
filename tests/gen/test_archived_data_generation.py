"""Selected tests for data generation and estimate of enforcement counts."""

import gc
import io
from pathlib import Path

import numpy as np
import pytest
from attrs import fields
from numpy.testing import assert_allclose

from mergeron import YAML
from mergeron import RECForm
from mergeron import UPPAggregator
from mergeron import zipfile
from mergeron.core import guidelines_boundaries as gbl
from mergeron.core.pseudorandom_numbers import seed_sequencer
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
ENFT_THRESHOLDS = gbl.GuidelinesStandards(2023).presumption
ENFT_REGIME = UPPTestRegime(INVResolution.ENFT, UPPAggregator.MIN, UPPAggregator.MIN)

tests_set = {
    # Test with uniform distribution (unrestricted shares), proportional recapture spec
    (
        SHRDistribution.UNI,
        RECForm.FIXED,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), proportional recapture spec
    (
        SHRDistribution.UNI,
        RECForm.FIXED,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # .i.i.d PCM values
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # .i.i.d PCM values
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # unimodal empirical PCM distribution, .i.i.d PCM values
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.EMPR_U,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # multimodal empirical PCM distribution, .i.i.d PCM values
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.EMPR_M,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # MNL-consistent PCM values
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # MNL-consistent PCM values, HSR filing requirement, SoP_NTH
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_TEN,
    ),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # MNL-consistent PCM values, HSR filing requirement, SoP_TEN
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.COST_SYM,
        HSRFilingTest.SoP_TEN,
    ),
    # Test with flat dirichlet, inside-out recapture spec, i.i.d. PCM values
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with flat dirichlet, inside-out recapture spec, MNL-consistent PCM values
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with flat dirichlet, inside-out recapture spec, MNL-consistent PCM values, SoP_NTH
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_NTH,
    ),
    # Test with flat dirichlet, inside-out recapture spec, MNL-consistent PCM values, SoP_TEN
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_TEN,
    ),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_RND,
        HSRFilingTest.NONE,
    ),
    # Test with flat dirichlet, outside-in recapture spec, i.i.d PCM values
    (
        SHRDistribution.DIR_FLAT,
        RECForm.OUTIN,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    # Test with unweighted flat dirichlet, inside-out recapture spec, i.i.d PCM values
    (
        SHRDistribution.DIR_FLAT_CONSTR,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    (
        SHRDistribution.DIR_ASYM,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
    (
        SHRDistribution.DIR_COND,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ),
}


@pytest.mark.parametrize("_test_spec", tests_set)
def test_markets_sampler(
    _test_spec: tuple[
        SHRDistribution,
        RECForm,
        PCMDistribution,
        PCMRestriction,
        PriceSpec,
        HSRFilingTest,
    ],
    _sample_size: int = 10**5,
    _nthreads: int = 16,
) -> None:
    """Data generation against stored samples."""
    _archive_name = "mergeron-market_sample-{}.zip".format(
        "-".join(_t.name for _t in _test_spec)
    )

    (
        _share_distribution,
        _recapture_form,
        _pcm_distribution,
        _pcm_restriction,
        _price_spec,
        _hsr_filing_test_type,
    ) = _test_spec

    # Reinitialize the seed sequence for each test run

    market_sample = MarketSample(
        hsr_filing_test_type=_hsr_filing_test_type,
        price_spec=_price_spec,
        pcm_spec=PCMSpec(_pcm_distribution, pcm_restriction=_pcm_restriction),
        share_spec=ShareSpec(
            _share_distribution,
            firm_counts_weights=(
                None
                if _share_distribution == SHRDistribution.UNI
                else (_nr := np.arange(1, 6)[::-1]) / _nr.sum()
            ),
            lower_bound=0.00,
            recapture_form=_recapture_form,
            recapture_rate=None if _recapture_form == RECForm.OUTIN else 0.80,
        ),
        sample_size=_sample_size,
        seed_data=seed_sequencer(len(fields(SeedSequenceData))),
        nthreads=_nthreads,
    )

    market_sample.generate_sample()

    _share_array = market_sample.dataset.share_array
    _aggr_purch_prob = market_sample.dataset.aggregate_purchase_probability

    # Test market shares
    if len(_share_array) != _sample_size:
        for _s in _test_spec:
            print(_s)
        print(len(_share_array), "=?", _sample_size)
        raise AssertionError(
            "Generated share array does not match specified sample size in length."
        )

    # Test diversion ratios
    _frmshr_array = _share_array[:, :2]
    _diversion_array = compute_merging_firm_diversion_ratios(
        market_sample.share_spec.recapture_form,
        market_sample.share_spec.recapture_rate,
        _frmshr_array,
        _aggr_purch_prob,
    )

    divr_assert_test = (
        (np.round(np.einsum("ij->i", _frmshr_array), 15) == 1)
        | (np.argmin(_frmshr_array, axis=1) == np.argmax(_diversion_array, axis=1))
    )[:, None]
    if not all(divr_assert_test):
        print(_frmshr_array, _diversion_array)
        raise ValueError(
            "{} {} {} {}".format(
                "Data construction fails tests:",
                "the index of min(s_1, s_2) must equal",
                "the index of max(d_12, d_21), for all draws.",
                "unless frmshr_array sums to 1.00.",
            )
        )

    if market_sample.share_spec.recapture_form != RECForm.FIXED:
        _purchase_prob_array = np.einsum("ij,ij->ij", _aggr_purch_prob, _share_array)

    if market_sample.share_spec.distribution.name != "UNI":
        try:
            assert_allclose(np.einsum("ij->", _share_array), float(_sample_size))
        except AssertionError as _err:
            print(np.einsum("ij->", _share_array), "=?", _sample_size)
            raise _err

        # Test aggregate purchase probability
        assert_allclose(
            np.einsum("ij->i", _purchase_prob_array)[:, None], _aggr_purch_prob
        )

        # Test diversion ratios
        _prod_idx = 0

        divratio_1j = np.divide(
            _purchase_prob_array, 1 - _purchase_prob_array[:, [_prod_idx]]
        )
        divratio_1j[:, _prod_idx] = 0

        assert_allclose(
            np.einsum("ij->i", divratio_1j)[:, None]
            + np.divide(1 - _aggr_purch_prob, 1 - _purchase_prob_array[:, [_prod_idx]]),
            np.ones_like(_aggr_purch_prob),
        )

    market_sample.compute_enforcement_counts(ENFT_THRESHOLDS, ENFT_REGIME)

    if market_sample.enforcement_counts.ByDelta[:, 1].sum() != _sample_size:
        raise AssertionError(
            "Generated sample size does not match benchmark specification."
        )

    object.__setattr__(market_sample, "dataset", None)
    # Save the market sample with generated data and estimated enforcement counts
    #  to a zip archive, if it does not already exist
    if not (_zap := TEST_DATA_DIR / _archive_name).is_file():
        with ZipFile(_zap, "w", 93) as _zaf:
            market_sample.to_archive(_zaf, save_dataset=True)

    if all((
        (_cds := market_sample.dataset) is None,
        (_cec := market_sample.enforcement_counts) is None,
    )):
        raise ValueError(
            "Market sample dataset does not exist. Run .generate_sample() and, "
            "when applicable, .compute_enforcement_counts() before proceeding."
        )

    # Load the market sample with generated data and estimated enforcement counts
    #  from the zip archive
    with ZipFile(_zap, "r") as _zaf:
        market_sample_bench = MarketSample.from_archive(_zaf, restore_dataset=True)

    # Roundtrip the market sample, no data, through YAML serialization
    # This just gives us a "bare" MarketSample object, i.e., without data, for testing
    with io.StringIO() as _yb:
        _ = YAML.dump(market_sample, _yb)
        _cms = YAML.load(_yb.getvalue())

    # Rebuild a bare MarketSample object from the deserialized YAML
    #   this demonstrates an alternative path to the same
    _tms = MarketSample(**{
        _a.name: getattr(market_sample_bench, _a.name)
        for _a in market_sample.__attrs_attrs__
        if _a.name not in {"dataset", "enforcement_counts"}
    })
    if _cms != _tms:
        print(_cms)
        print(_tms)
        raise AssertionError

    if _cec != market_sample_bench.enforcement_counts:
        for _f in ("ByFirmCount", "ByDelta", "ByHHIandDelta"):
            print(f'Enforcement counts, "{_f}" differ between test and generated data.')
            print(getattr(_cec, _f))
            print(getattr(market_sample_bench.enforcement_counts, _f))
        raise AssertionError

    del market_sample
    gc.collect()
