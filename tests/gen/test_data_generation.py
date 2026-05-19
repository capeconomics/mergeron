"""Test accuracy of data generation for selected market configurations."""

import gc

import numpy as np
import pytest
from attrs import fields
from numpy.testing import assert_allclose

from mergeron import ArrayDouble
from mergeron import RECForm
from mergeron.core.pseudorandom_numbers import seed_sequencer
from mergeron.gen import HSRFilingTest
from mergeron.gen import PCMDistribution
from mergeron.gen import PCMRestriction
from mergeron.gen import PCMSpec
from mergeron.gen import PriceSpec
from mergeron.gen import SeedSequenceData
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution
from mergeron.gen.data_generation import MarketSample
from mergeron.gen.data_generation_functions import compute_all_firm_diversion_ratios
from mergeron.gen.data_generation_functions import compute_merging_firm_diversion_ratios

bench_vals_dict = {
    (
        SHRDistribution.UNI,
        RECForm.FIXED,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3333082993693158302406231996,
        0.3999541207553178878697508480,
        0.1666179396597210482511286500,
    ]),
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3333082993693158302406231996,
        0.3737269898086484354315928158,
        0.1666179396597210482511286500,
    ]),
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.EMPR_M,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3333082993693158302406231996,
        0.3737269898086484354315928158,
        0.1666179396597210482511286500,
    ]),
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3249672413552518457358075921,
        0.3669546247222009172084256079,
        0.1672788653047521389982676965,
    ]),
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.EMPR_M,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_TEN,
    ): np.array([
        0.3247588944672321620288357735,
        0.3671897828587237788688923956,
        0.1681387463869197518295806049,
    ]),
    (
        SHRDistribution.UNI,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_TEN,
    ): np.array([
        0.3249672413552518457358075921,
        0.3669546247222009172084256079,
        0.1672788653047521389982676965,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.EMPR_U,
        PCMRestriction.IID,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.EMPR_M,
        PCMRestriction.IID,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.COST_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_RND,
        HSRFilingTest.NONE,
    ): np.array([
        0.3617328692797409650516726742,
        0.4348189799830503576849594083,
        0.2035778789175776148923091569,
        0.7121643300181268454451810612,
        0.3084812285665262110434525766,
        3.2312509999999998733244410687,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3511338070055080584630502472,
        0.4231200328825999723569850630,
        0.2015443441173271343913597775,
        0.6825885668273996564892058814,
        0.2959645162466362777742290291,
        3.3497770000000000045758952183,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_NTH,
    ): np.array([
        0.3511338070055080584630502472,
        0.4231200328825999723569850630,
        0.2015443441173271343913597775,
        0.6825885668273996564892058814,
        0.2959645162466362777742290291,
        3.3497770000000000045758952183,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.MNL,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.SoP_TEN,
    ): np.array([
        0.3511338070055080584630502472,
        0.4231200328825999723569850630,
        0.2015443441173271343913597775,
        0.6825885668273996564892058814,
        0.2959645162466362777742290291,
        3.3497770000000000045758952183,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT,
        RECForm.OUTIN,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431559621940251658678278091,
        0.3429485279699741040460025943,
        0.1873937722436013197935267272,
        0.6862932774682282133227317900,
        0.3432075727904080331143177318,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_FLAT_CONSTR,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3431813076324266575944932356,
        0.4071457151001731666895011585,
        0.1873215766800016279791663010,
        0.6862912883714991085781775837,
        0.3436262877700239637412948923,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_ASYM,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.3432548580631789580941415352,
        0.4269078053192427724127355759,
        0.2192258955894157423927026684,
        0.6517924708314303261502686837,
        0.3432622529591694560124892632,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
    (
        SHRDistribution.DIR_COND,
        RECForm.INOUT,
        PCMDistribution.UNI,
        PCMRestriction.IID,
        PriceSpec.PRICE_SYM,
        HSRFilingTest.NONE,
    ): np.array([
        0.4443975155403350751903701621,
        0.5976399440631765536124930804,
        0.3372898908431334330515483089,
        0.8354255349427489596436657848,
        0.2377506190300675559257115310,
        3.3347199999999999064925759740,
        6.0000000000000000000000000000,
    ]),
}


@pytest.mark.parametrize("_test_parms, _test_array", tuple(bench_vals_dict.items()))
def test_markets_sampler(
    _test_parms: tuple[
        SHRDistribution,
        RECForm,
        PCMDistribution,
        PCMRestriction,
        PriceSpec,
        HSRFilingTest,
    ],
    _test_array: ArrayDouble,
    _tcount: int = 10**6,
    _nthreads: int = 16,
) -> None:
    """Test generation of market samples with various parameters."""
    (
        share_distribution,
        recapture_form,
        pcm_distribution,
        pcm_restriction,
        price_spec,
        hsr_filing_test_type,
    ) = _test_parms

    market_sample = MarketSample(
        share_spec=ShareSpec(
            share_distribution,
            firm_counts_weights=(
                None
                if share_distribution == SHRDistribution.UNI
                else (_nr := np.arange(1, 6)[::-1]) / _nr.sum()
            ),
            lower_bound=0.0,
            recapture_form=recapture_form,
            recapture_rate=None if recapture_form == RECForm.OUTIN else 0.80,
        ),
        pcm_spec=PCMSpec(
            distribution=pcm_distribution, pcm_restriction=pcm_restriction
        ),
        price_spec=price_spec,
        hsr_filing_test_type=hsr_filing_test_type,
        sample_size=_tcount,
        seed_data=seed_sequencer(len(fields(SeedSequenceData))),
        nthreads=1,
    )

    if market_sample.dataset is None:
        market_sample.generate_sample()

    _data_to_test = market_sample.dataset
    _market_shares = _data_to_test.shares
    _fcounts_test = np.einsum("ij->i", _market_shares > 0, dtype="<u8")[:, None]

    # Test market share array
    if not len(_market_shares) == _tcount:
        raise AssertionError(
            "DATA GENERATION ERROR: {} {}".format(
                "Generation of sample shares is inconsistent:",
                "Number of draws does not match specified sample size.",
            )
        )

    if "DIR" in share_distribution.name:
        if (_iss := np.round(np.einsum("ij->", _market_shares))) != _tcount:
            print(_iss, _tcount, len(_market_shares))
            raise AssertionError(
                "DATA GENERATION ERROR: {} {} {}".format(
                    "Generation of sample shares is inconsistent:",
                    "array of drawn shares must sum to the number of draws",
                    "i.e., the sample size, which condition is not met.",
                )
            )

        # Test fcounts
        _fcounts_keys, _fcounts_counts = np.unique(_fcounts_test, return_counts=True)
        if not (
            (
                _fcounts_keys
                == 2 + np.arange(len(market_sample.share_spec.firm_counts_weights))
            ).all()
        ):
            print(
                "keys generated:",
                _fcounts_keys,
                "counts generated:",
                _fcounts_counts,
                "keys expected:",
                2 + np.arange(len(market_sample.share_spec.firm_counts_weights)),
                sep="\n",
            )
            raise AssertionError(
                "DATA GENERATION ERROR: {} {} {}".format(
                    "Generation of sample shares is inconsistent:",
                    "In each draw, non-zero shares must number the firm counts implied by",
                    "the specified firm-count weights.",
                )
            )

        # Test fcounts
        _firm_count_weights = market_sample.share_spec.firm_counts_weights
        if (
            (
                market_sample.pcm_spec.pcm_restriction != PCMRestriction.MNL
                or market_sample.price_spec == PriceSpec.COST_SYM
            )
            and not (market_sample.hmt_flag and market_sample.hmt_flag.recompute_shares)
            and not np.allclose(
                _fcw_test := _fcounts_counts / _fcounts_test.size,
                _firm_count_weights,
                atol=1 / np.sqrt(_tcount),
            )
        ):
            print(_fcw_test, _firm_count_weights, sep="\n")
            raise AssertionError(
                "DATA GENERATION ERROR: {} {} {}".format(
                    "Generation of sample shares is inconsistent:",
                    "array of drawn shares must exhibit firm counts",
                    "in proportion to firm-count weights.",
                )
            )

    _aggr_purch_prob = market_sample.dataset.aggregate_choice_probability
    # Test diversion ratios
    _mrgng_firm_shares = _market_shares[:, :2]
    _diversion_ratios = compute_merging_firm_diversion_ratios(
        market_sample.share_spec.recapture_form,
        market_sample.share_spec.recapture_rate,
        _mrgng_firm_shares,
        _aggr_purch_prob,
    )
    divr_assert_test = (
        (np.round(np.einsum("ij->i", _mrgng_firm_shares), 15) == 1)
        | (
            np.argmin(_mrgng_firm_shares, axis=1)
            == np.argmax(_diversion_ratios, axis=1)
        )
    )[:, None]
    if not all(divr_assert_test):
        print(_mrgng_firm_shares, _diversion_ratios)
        raise ValueError(
            "{} {} {} {}".format(
                "Data construction fails tests:",
                "the index of min(s_1, s_2) must equal",
                "the index of max(d_12, d_21), for all draws.",
                "unless frmshr_array sums to 1.00.",
            )
        )

    _diversion_ratios_allfirm = compute_all_firm_diversion_ratios(
        market_sample.share_spec.recapture_form,
        market_sample.share_spec.recapture_rate,
        _market_shares,
        _aggr_purch_prob,
    )
    if not np.allclose(_diversion_ratios, _diversion_ratios_allfirm[:, [1, 0], [0, 1]]):
        print(_diversion_ratios, _diversion_ratios_allfirm[:, [1, 0], [0, 1]])
        raise ValueError(
            "{} {} {}".format(
                "Diversion ratio computation fails tests:",
                "merging-firm diversion ratios calculated directly",
                "must match those extracted from all-firm diversion ratios.",
            )
        )

    # Test market share array and _aggr_purch_prob:
    if (
        market_sample.share_spec.recapture_form != RECForm.FIXED
        and market_sample.share_spec.distribution != SHRDistribution.UNI
    ):
        _choice_probabilities = np.einsum("ij,ij->ij", _aggr_purch_prob, _market_shares)
        try:
            np.testing.assert_array_almost_equal(
                _choice_probabilities.sum(axis=1, keepdims=True), _aggr_purch_prob
            )
        except AssertionError as _e:
            raise ValueError(
                "DATA GENERATION ERROR: {}".format(
                    "Choice probabilities don't add to aggregate choice probabilities."
                )
            ) from _e

        if share_distribution != SHRDistribution.UNI:
            assert_allclose(
                np.einsum("ijk->ik", _diversion_ratios_allfirm)
                + (1 - _aggr_purch_prob) / (1 - _choice_probabilities),
                1.0,
            )

    # Test HHI
    _nth_firm_share = np.take_along_axis(_market_shares, _fcounts_test - 1, axis=1)
    _hhi_delta = np.einsum("ij,ij->i", _mrgng_firm_shares, _mrgng_firm_shares[:, ::-1])[
        :, None
    ]
    _hhi_post = (
        _hhi_delta + np.einsum("ij,ij->i", _market_shares, _market_shares)[:, None]
    )

    if _data_to_test is not None:
        array_to_test = np.array(
            [_market_shares[:, :2].mean(), _diversion_ratios.mean(), _hhi_delta.mean()]
            + (
                []
                if share_distribution == SHRDistribution.UNI
                else [
                    _hhi_post.mean(),
                    _nth_firm_share.mean(),
                    _fcounts_test.mean(),
                    _fcounts_test.max(),
                ]
            )
        )

    print(
        share_distribution,
        recapture_form,
        pcm_distribution,
        pcm_restriction,
        price_spec,
        hsr_filing_test_type,
        "\n",
        f"Test with {_tcount:,d} draws",
        "\n",
        repr(array_to_test),
    )

    assert_allclose(array_to_test, _test_array, atol=1e-15, rtol=1e-15)
    del market_sample, array_to_test
    gc.collect()
