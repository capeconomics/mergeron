"""Test methods/functions for analyzing  boundaries for Guidelines standards."""

import gc
from collections.abc import Sequence

import pytest
from mpmath import mp  # type: ignore
from mpmath import mpf  # type: ignore
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

import mergeron.core.guidelines_boundaries as gbl
import mergeron.core.guidelines_boundary_functions as gbf
import mergeron.perks.guidelines_boundary_functions_extra as gbx
from mergeron import RECForm
from mergeron import UPPAggregator

gval_print_format_str = "g_val = {}; m_val = {}; {} =? {}"


@pytest.mark.parametrize(
    "_num, _frac, _mode, _test_val",
    (
        (0.0624, 0.005, "ROUND_HALF_UP", 0.06),
        (8.8888, 0.50, "ROUND_DOWN", 8.5),
        (8.8888, 0.50, "ROUND_HALF_UP", 9.0),
    ),
)
def test_round_cust(_num: float, _frac: float, _mode: str, _test_val: float) -> None:
    assert_equal(
        float(gbf.round_cust(_num, frac=_frac, rounding_mode=_mode)), _test_val
    )


def test_round_cust_to_fp_aproximation_error() -> None:
    # Difference below is due to floating-point approx. error
    assert_almost_equal(
        float(gbf.round_cust(12.35, frac=0.05, rounding_mode="ROUND_DOWN")),
        12.35,
        decimal=12,
    )


def test_lerp() -> None:
    assert_equal(gbf.lerp(1, 3, 0.25), 1.5)


@pytest.mark.parametrize(
    "_dhv, _rbar, _test_val",
    (
        (0.01, 4 / 5, 0.06),
        (0.02, 4 / 5, 0.09),
        (0.005, 6 / 7, 0.045),
        (0.01, 6 / 7, 0.065),
    ),
)
def test_guppi_from_delta(_dhv: float, _rbar: float, _test_val: float) -> None:
    assert_equal(float(gbl.guppi_from_delta(_dhv, r_bar=_rbar)), _test_val)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _test_val", ((0.06, 1.00, 4 / 5, 0.070), (0.09, 0.40, 0.9, 0.200))
)
def test_share_from_guppi(_gv: float, _mv: float, _rv: float, _test_val: float) -> None:
    assert_equal(float(gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv)), _test_val)


@pytest.mark.parametrize(
    "_test_parms, _test_val",
    tuple(zip(((), (0.045, 1.00, 6 / 7)), (0.075, 5 / 95), strict=True)),
)
def test_benchmark_shrratio(_test_parms: Sequence[float], _test_val: float) -> None:
    if _test_parms:
        gv_, mv_, rv_ = _test_parms
        ts_ = gbl.critical_diversion_share(gv_, m_star=mv_, r_bar=rv_)
    else:
        ts_ = gbl.critical_diversion_share()
    assert_equal(gbf.round_cust(ts_), gbf.round_cust(_test_val))


def print_done() -> None:
    print("... done.")


_dh_tuple = ((0.01, 0.03147), (0.02, 0.05595), (0.08, 0.16709))


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_dh_area(_dhv: float, _dha: float) -> None:
    print(f"Testing gbl.dh_area() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(
            _av := gbf.dh_area(_dhv, dps=10), _tv := round(gbx.dh_area_quad(_dhv), 10)
        )
    except AssertionError as _err:
        print(_av, "=?", _tv, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_hhi_delta_boundary_dha(_dhv: float, _dha: float) -> None:
    ts_ = gbf.hhi_delta_boundary(_dhv).area
    print(f"Testing gbl.hhi_delta_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(ts_, _dha)
    except AssertionError as _err:
        print(gbf.dh_area(_dhv), "=?", ts_, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.01, 0.02, 0.08))
def test_hhi_delta_boundary(_dhv: float) -> None:
    ts_ = gbf.hhi_delta_boundary(_dhv).area
    print(f"Testing gbl.hhi_delta_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(ts_, round(gbf.dh_area(_dhv), 5))
    except AssertionError as _err:
        print(gbf.dh_area(_dhv), "=?", ts_, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.02, 0.0625, 0.16))
def test_combined_share_boundary(_dhv: float) -> None:
    assert_equal(gbf.combined_share_boundary(mp.sqrt(_dhv)).area, _dhv / 2)


@pytest.mark.parametrize("_dhv", (0.02, 0.03125, 0.08))
def test_hhi_pre_contrib_boundary(_dhv: float) -> None:
    assert_equal(
        gbf.hhi_pre_contrib_boundary(_dhv).area, round(mp.pi * mpf(f"{_dhv}") / 4, 5)
    )


# Test Guidleines threholds
@pytest.mark.parametrize(
    ("_pub_year", "_std", "_test_val"),
    (
        (
            1992,
            "safeharbor",
            gbl.MGThresholds(
                delta=0.005,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.045),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2010,
            "safeharbor",
            gbl.MGThresholds(
                delta=0.01,
                fc=4,
                rec=(_rv := 0.8),
                guppi=(_g := 0.06),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2023,
            "safeharbor",
            gbl.MGThresholds(
                delta=0.01,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.065),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            1992,
            "presumption",
            gbl.MGThresholds(
                delta=0.01,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.045),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2010,
            "presumption",
            gbl.MGThresholds(
                delta=0.02,
                fc=4,
                rec=(_rv := 0.8),
                guppi=(_g := 0.06),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2023,
            "presumption",
            gbl.MGThresholds(
                delta=0.01,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.065),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            1992,
            "imputed_presumption",
            gbl.MGThresholds(
                delta=0.04081632653061224,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.045),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2010,
            "imputed_presumption",
            gbl.MGThresholds(
                delta=0.03125,
                fc=4,
                rec=(_rv := 0.8),
                guppi=(_g := 0.06),
                dr=(1 - _rv) / 2,
                cmcr=_g,
                ipr=_g,
            ),
        ),
        (
            2023,
            "imputed_presumption",
            gbl.MGThresholds(
                delta=0.04081632653061224,
                fc=6,
                rec=(_rv := 0.85),
                guppi=(_g := 0.065),
                dr=1 - _rv,
                cmcr=_g,
                ipr=_g,
            ),
        ),
    ),
)
def test_guidelines_thresholds(
    _pub_year: int, _std: str, _test_val: gbl.MGThresholds
) -> None:
    ts_ = gbl.GuidelinesStandards(_pub_year)
    assert_equal(getattr(ts_, _std), _test_val)


# Test boudnary functions
@pytest.mark.parametrize(
    "_dhv, _tv",
    tuple(
        zip(
            ((0.06, 1.00), (0.06, 0.67), (0.06, 0.30)),
            (0.0052325581, 0.0112691576, 0.05),
            strict=True,
        )
    ),
)
def test_diversion_ratio_boundary_at_max(_dhv: tuple[float, float], _tv: float) -> None:
    _test_area = gbl.DiversionBoundary(
        gbl.critical_diversion_share(_dhv[0], m_star=_dhv[1], r_bar=1.00),
        0.80,
        aggregator=UPPAggregator.MAX,
        precision=10,
    ).area
    assert_equal(_test_area, _tv)


@pytest.mark.parametrize(
    "_dhv, _tv",
    tuple(
        zip(
            ((0.06, 1.00), (0.06, 0.67), (0.06, 0.30)),
            (0.0052325581, 0.0112691576, 0.05),
            strict=True,
        )
    ),
)
def test_diversion_share_boundary_max(_dhv: tuple[float, float], _tv: float) -> None:
    _test_area = gbf.diversion_share_boundary_max(
        gbl.critical_diversion_share(_dhv[0], m_star=_dhv[1], r_bar=0.80)
    ).area
    assert_equal(_test_area, _tv)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _recapture_form",
    (
        (0.06, 1.00, 0.8, RECForm.FIXED),
        (0.06, 0.67, 0.8, RECForm.FIXED),
        (0.06, 0.30, 0.8, RECForm.FIXED),
    ),
)
def test_diversion_ratio_boundary_at_min(
    _gv: float, _mv: float, _rv: float, _recapture_form: RECForm
) -> None:
    _test_area = gbl.DiversionBoundary(
        gbl.critical_diversion_share(_gv, m_star=_mv, r_bar=1.00),
        _rv,
        aggregator=UPPAggregator.MIN,
        recapture_form=_recapture_form,
        precision=10,
    ).area
    assert_equal(
        gbf.round_cust(_test_area), gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv)
    )


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _recapture_form",
    (
        (0.06, 1.00, 0.8, "proportional"),
        (0.06, 0.67, 0.8, "proportional"),
        (0.06, 0.30, 0.8, "proportional"),
    ),
)
def test_diversion_share_boundary_min(
    _gv: float, _mv: float, _rv: float, _recapture_form: str
) -> None:
    assert_equal(
        gbf.round_cust(
            gbf.diversion_share_boundary_min(
                gbl.critical_diversion_share(_gv, m_star=_mv, r_bar=_rv),
                _rv,
                recapture_form=_recapture_form,
            ).area
        ),
        gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv),
    )


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, UPPAggregator.OSA, RECForm.FIXED, 0.05111),
        (0.06, 0.67, UPPAggregator.OSA, RECForm.FIXED, 0.07726),
        (0.06, 0.3, UPPAggregator.OSA, RECForm.FIXED, 0.17098),
        (0.06, 1.0, UPPAggregator.CPA, RECForm.FIXED, 0.00658),
        (0.06, 0.67, UPPAggregator.CPA, RECForm.FIXED, 0.01409),
        (0.06, 0.3, UPPAggregator.CPA, RECForm.FIXED, 0.0606),
        (0.06, 1.0, UPPAggregator.AVG, RECForm.FIXED, 0.0102),
        (0.06, 0.67, UPPAggregator.AVG, RECForm.FIXED, 0.02169),
        (0.06, 0.3, UPPAggregator.AVG, RECForm.FIXED, 0.09123),
        (0.06, 1.0, UPPAggregator.AVG, RECForm.INOUT, 0.01026),
        (0.06, 0.67, UPPAggregator.AVG, RECForm.INOUT, 0.02187),
        (0.06, 0.3, UPPAggregator.AVG, RECForm.INOUT, 0.09323),
    ),
)
def test_diversion_ratio_boundary(_tvl: tuple[float, float, str, str, float]) -> None:
    ts_ = gbl.DiversionBoundary(
        gbl.critical_diversion_share(_tvl[0], m_star=_tvl[1], r_bar=1.0),
        0.80,
        aggregator=_tvl[2],  # type: ignore
        recapture_form=_tvl[3],  # type: ignore
    ).area
    print("Test gbl.DiversionBoundary(): ", end="")
    try:
        assert_equal(ts_, _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; meanf = {}; {}".format(*_tvl),
            "=?",
            ts_,
            end="",
        )
        raise _err
    print_done()


diversion_share_boundary_wtd_avg_test_values = (
    (0.06, 1.0, "own-share", "arithmetic mean", "proportional", 0.05111),
    (0.06, 0.67, "own-share", "arithmetic mean", "proportional", 0.07726),
    (0.06, 0.3, "own-share", "arithmetic mean", "proportional", 0.17098),
    (
        0.06,
        1.0,
        "cross-product-share",
        "arithmetic mean",
        "proportional",
        0.00658,
    ),  # with _manualroot: 0.00658, with _autoroot: 0.0066),
    (0.06, 0.67, "cross-product-share", "arithmetic mean", "proportional", 0.01409),
    (0.06, 0.3, "cross-product-share", "arithmetic mean", "proportional", 0.0606),
    (0.06, 1.0, None, "arithmetic mean", "proportional", 0.0102),
    (0.06, 0.67, None, "arithmetic mean", "proportional", 0.02169),
    (0.06, 0.3, None, "arithmetic mean", "proportional", 0.09123),
    (
        0.06,
        1.0,
        None,
        "arithmetic mean",
        "inside-out",
        0.01026,
    ),  # with _manualroot: 0.01026, with _autoroot: 0.01025),
    (0.06, 0.67, None, "arithmetic mean", "inside-out", 0.02187),
    (0.06, 0.3, None, "arithmetic mean", "inside-out", 0.09323),
)


@pytest.mark.parametrize("_tvl", diversion_share_boundary_wtd_avg_test_values)
def test_diversion_share_boundary_wtd_avg(
    _tvl: tuple[float, float, str, str, float],
) -> None:
    """Test the underlying boundary function directly."""
    ts_ = gbf.diversion_share_boundary_wtd_avg(
        gbl.critical_diversion_share(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],
        aggregator=_tvl[3],
        recapture_form=_tvl[4],
    ).area
    print("Test gbf.diversion_share_boundary_wtd_avg(): ", end="")
    try:
        assert_equal(ts_, _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; meanf = {}; {}".format(*_tvl),
            "=?",
            ts_,
            end="",
        )
        raise _err
    print_done()


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "inside-out", 0.01026),
        (0.06, 0.67, "inside-out", 0.02187),
        (0.06, 0.3, "inside-out", 0.09323),
        (0.06, 1.0, "proportional", 0.0102),
        (0.06, 0.67, "proportional", 0.02169),
        (0.06, 0.3, "proportional", 0.09123),
    ),
)
def test_diversion_share_boundary_xact_avg(
    _tvl: tuple[float, float, str, float],
) -> None:
    ts_ = gbf.diversion_share_boundary_xact_avg(
        gbl.critical_diversion_share(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        recapture_form=_tvl[2],
    ).area
    print("Test gbl.diversion_share_boundary_xact_avg(): ", end="")
    try:
        assert_equal(ts_, _tvl[-1])
    except AssertionError as _err:
        print(gval_print_format_str.format(*_tvl, ts_), end="")
        raise _err
    print_done()


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "own-share", "proportional", 0.05109304376203811),
        (0.06, 0.67, "own-share", "proportional", 0.07725625014417319),
        (0.06, 0.3, "own-share", "proportional", 0.17095706110994796),
        (0.06, 1.0, "cross-product-share", "proportional", 0.006600706829415734),
        (0.06, 0.67, "cross-product-share", "proportional", 0.01409071869102121),
        (0.06, 0.3, "cross-product-share", "proportional", 0.0606003596099902),
        (0.06, 1.0, None, "proportional", 0.010202305341592574),
        (0.06, 0.67, None, "proportional", 0.021688811233381202),
        (0.06, 0.3, None, "proportional", 0.09122636371611485),
        (0.06, 1.0, None, "inside-out", 0.010256625940293616),
        (0.06, 0.67, None, "inside-out", 0.021867653765402204),
        (0.06, 0.3, None, "inside-out", 0.093231884646380737814431487),
    ),
)
def test_diversion_share_boundary_qdtr_wtd_avg(
    _tvl: tuple[float, float, str, str, float],
) -> None:
    ts_ = gbx.diversion_share_boundary_qdtr_wtd_avg(
        gbl.critical_diversion_share(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],
        recapture_form=_tvl[3],
    ).area
    print("Test gbx.diversion_share_boundary_qdtr_wtd_avg(): ", end="")
    try:
        assert_equal(ts_, _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; recapture_form = {}; {}".format(*_tvl),
            "=?",
            ts_,
            end="",
        )
        raise _err
    print_done()


@pytest.mark.parametrize("_tvl", diversion_share_boundary_wtd_avg_test_values)
def test_diversion_share_boundary_distance(
    _tvl: tuple[float, float, str, str, float],
) -> None:
    ts_ = gbx.diversion_share_boundary_distance(
        gbl.critical_diversion_share(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],
        aggregator=_tvl[3],
        recapture_form=_tvl[4],
    ).area
    print("Test gbx.test_diversion_share_boundary_distance(): ", end="")
    try:
        assert_equal(ts_, _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; meanf = {}; {}".format(*_tvl),
            "=?",
            ts_,
            end="",
        )
        raise _err
    print_done()


gc.collect()
