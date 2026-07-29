"""
Specialized methods for defining and analyzing boundaries for Guidelines standards.

These methods (functions) provide rely on scipy of sympy for core computations,
and may provide improved precision than core functions, but tend to have
poor performance

"""

from __future__ import annotations

from typing import Literal

import numpy as np
from mpmath import mp
from mpmath import mpf
from scipy.spatial.distance import minkowski
from sympy import lambdify
from sympy import simplify
from sympy import solve
from sympy import symbols

from .. import DEFAULT_REC
from .. import VERSION
from .. import MPFloat
from ..core import GuidelinesBoundary
from ..core import guidelines_boundary_functions as gbf
from . import GuidelinesBoundaryCallable

__version__ = VERSION


mp.dps = 32
mp.trap_complex = True


def dh_area_quad(_dh_val: float = 0.01, /) -> float:
    """
    Area under the ΔHHI boundary.

    When the given ΔHHI bound matches a Guidelines safeharbor,
    the area under the boundary is half the intrinsic clearance rate
    for the ΔHHI safeharbor.

    Parameters
    ----------
    _dh_val
        Merging-firms' ΔHHI bound.

    Returns
    -------
        Area under ΔHHI boundary.

    """
    _dh_val = mpf(f"{_dh_val}")
    _s_naught = (1 - mp.sqrt(1 - 2 * _dh_val)) / 2

    return float(
        mp.nstr(
            _s_naught
            + mp.quad(lambda x: _dh_val / (2 * x), [_s_naught, 1 - _s_naught]),
            mp.prec,
        )
    )


def hhi_delta_boundary_qdtr(_dh_val: float = 0.01, /) -> GuidelinesBoundaryCallable:
    """
    Generate the list of share combination on the ΔHHI boundary.

    Parameters
    ----------
    _dh_val:
        Merging-firms' ΔHHI bound.

    Returns
    -------
        Callable to generate array of share-pairs, area under boundary.

    """
    _dh_val = mpf(f"{_dh_val}")

    _s_1, _s_2 = symbols("s_1, s_2", positive=True)

    _hhi_eqn = _s_2 - _dh_val / (2 * _s_1)

    hhi_bdry = solve(_hhi_eqn, _s_2)[0]
    s_nought = float(solve(_hhi_eqn.subs({_s_2: 1 - _s_1}), _s_1)[0])

    hhi_bdry_area = float(
        mp.nstr(
            2
            * (
                s_nought
                + mp.quad(lambdify(_s_1, hhi_bdry, "mpmath"), (s_nought, 1 - s_nought))
            ),
            mp.prec,
        )
    )

    return GuidelinesBoundaryCallable(
        lambdify(_s_1, hhi_bdry, "numpy"), hhi_bdry_area, s_nought
    )


def diversion_share_boundary_qdtr_wtd_avg(
    _delta_star: float | MPFloat = 0.075,
    _r_val: float | MPFloat = DEFAULT_REC,
    /,
    *,
    weighting: Literal["own-share", "cross-product-share"] | None = "own-share",
    recapture_form: Literal["inside-out", "fixed"] = "inside-out",
) -> GuidelinesBoundaryCallable:
    R"""
    Share combinations for the share-weighted average share-ratio boundary.

    Parameters
    ----------
    _delta_star
        Diversion share, :math:`\overline{d} / \overline{r}` or :math:`\overline{g} / (m^* \cdot \overline{r})`.
    _r_val
        Recapture rate.
    weighting
        Whether "own-share" or "cross-product-share" (or None for simple, unweighted average)
    recapture_form
        Whether recapture rate is share-proportional ("inside-out") or has fixed
        value for both merging firms ("fixed").

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _delta_star, _r_val = (mpf(_v) for _v in (f"{_delta_star}", f"{_r_val}"))
    _s_mid = _delta_star / (1 + _delta_star)
    s_naught = mpf("0.0")

    s_1, s_2 = symbols("s_1:3", positive=True)

    match weighting:
        case "own-share":
            _bdry_eqn = (
                s_1 * s_2 / (1 - s_1)
                + s_2
                * s_1
                / (
                    (1 - (_r_val * s_2 + (1 - _r_val) * s_1))
                    if recapture_form == "inside-out"
                    else (1 - s_2)
                )
                - (s_1 + s_2) * _delta_star
            )

            _bdry_func = solve(_bdry_eqn, s_2)[0]

            if recapture_form == "inside-out":
                s_naught = mpf(solve(simplify(_bdry_eqn.subs({s_2: 1 - s_1})), s_1)[0])

            bdry_area = mp.fmul(
                "2",
                (
                    s_naught
                    + mp.quad(lambdify(s_1, _bdry_func, "mpmath"), (s_naught, _s_mid))
                ),
            ) - (_s_mid**2 + s_naught**2)

        case "cross-product-share":
            mp.trap_complex = False
            d_star = symbols("d", positive=True)
            _bdry_eqn = (
                s_2 * s_2 / (1 - s_1)
                + s_1
                * s_1
                / (
                    (1 - (_r_val * s_2 + (1 - _r_val) * s_1))
                    if recapture_form == "inside-out"
                    else (1 - s_2)
                )
                - (s_1 + s_2) * d_star
            )

            _bdry_func = solve(_bdry_eqn, s_2)[1]
            bdry_area = (
                mp.fmul(
                    "2",
                    mp.quad(
                        lambdify(s_1, _bdry_func.subs({d_star: _delta_star}), "mpmath"),
                        (0, _s_mid),
                    ).real,
                )
                - _s_mid**2
            )

        case _:
            _bdry_eqn = (
                1 / 2 * s_2 / (1 - s_1)
                + 1
                / 2
                * s_1
                / (
                    (1 - (_r_val * s_2 + (1 - _r_val) * s_1))
                    if recapture_form == "inside-out"
                    else (1 - s_2)
                )
                - _delta_star
            )

            _bdry_func = solve(_bdry_eqn, s_2)[0]
            bdry_area = (
                mp.fmul("2", mp.quad(lambdify(s_1, _bdry_func, "mpmath"), (0, _s_mid)))
                - _s_mid**2
            )

    return GuidelinesBoundaryCallable(
        lambdify(s_1, _bdry_func, "numpy"), float(bdry_area), float(s_naught)
    )


def diversion_share_boundary_distance(
    _delta_star: float | MPFloat = 0.075,
    _r_val: float = DEFAULT_REC,
    /,
    *,
    aggregator: Literal["arithmetic mean", "distance"] = "arithmetic mean",
    weighting: Literal["own-share", "cross-product-share"] | None = "own-share",
    recapture_form: Literal["inside-out", "fixed"] = "inside-out",
    dps: int = 5,
) -> GuidelinesBoundary:
    R"""
    Share combinations for the share-ratio boundaries using various aggregators.

    Reimplements the arithmetic-averages and distance estimations from function,
    `.core.guidelines_boundary_functions.diversion_share_boundary_wtd_avg` but uses the Minkowski-distance function,
    `scipy.spatial.distance.minkowski` for all aggregators. This reimplementation
    is useful for testing the output of `.core.guidelines_boundary_functions.diversion_share_boundary_wtd_avg`
    but runs considerably slower.

    Parameters
    ----------
    _delta_star
        Diversion share, :math:`\overline{d} / \overline{r}` or :math:`\overline{g} / (m^* \cdot \overline{r})`.
    _r_val
        Recapture rate.
    aggregator
        Whether "arithmetic mean" or "distance".
    weighting
        Whether "own-share" or "cross-product-share".
    recapture_form
        Whether recapture rate is share-proportional ("inside-out") or has fixed
        value for both merging firms ("fixed").
    dps
        Number of decimal places for rounding returned shares and area.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _delta_star, _r_val = (mpf(_v) for _v in (f"{_delta_star}", f"{_r_val}"))

    # parameters for iteration
    _s_mid = mp.fdiv(_delta_star, 1 + _delta_star)
    _weights_base = (mpf("0.5"),) * 2
    _bdry_step_sz = mp.power(10, -dps)
    _theta = _bdry_step_sz * (10 if weighting == "cross-product-share" else 1)

    # initial conditions
    bdry_points = [(_s_mid, _s_mid)]
    s_1_pre, s_2_pre = _s_mid, _s_mid
    s_2_oddval, s_2_oddsum, s_2_evnsum = True, 0.0, 0.0
    for s_1 in mp.arange(_s_mid - _bdry_step_sz, 0, -_bdry_step_sz):
        # The wtd. avg. GUPPI is not always convex to the origin, so we
        #   increment s_2 after each iteration in which our algorithm
        #   finds (s1, s2) on the boundary
        s_2 = s_2_pre * (1 + _theta)

        if (s_1 + s_2) > mpf("0.99875"):
            # 1: # inaccuracy at 3-9s and up
            break

        while True:
            de_1 = s_2 / (1 - s_1)
            de_2 = (
                s_1 / (1 - gbf.lerp(s_1, s_2, _r_val))
                if recapture_form == "inside-out"
                else s_1 / (1 - s_2)
            )

            weights_i = (
                (
                    _w := mp.fdiv(
                        s_2 if weighting == "cross-product-share" else s_1, s_1 + s_2
                    ),
                    1 - _w,
                )
                if weighting
                else _weights_base
            )

            match aggregator:
                case "arithmetic mean":
                    delta_test = minkowski((de_1, de_2), (0.0, 0.0), p=1, w=weights_i)
                case "distance":
                    delta_test = minkowski((de_1, de_2), (0.0, 0.0), p=2, w=weights_i)

            _test_flag, _incr_decr = (
                (delta_test > _delta_star, -1)
                if weighting == "cross-product-share"
                else (delta_test < _delta_star, 1)
            )

            if _test_flag:
                s_2 += _incr_decr * _bdry_step_sz
            else:
                break

        # Build-up boundary points
        bdry_points.append((s_1, s_2))

        # Build up area terms
        s_2_oddsum += s_2 if s_2_oddval else 0
        s_2_evnsum += s_2 if not s_2_oddval else 0
        s_2_oddval = not s_2_oddval

        # Hold share points
        s_2_pre = s_2
        s_1_pre = s_1

    if s_2_oddval:
        s_2_evnsum -= s_2_pre
    else:
        s_2_oddsum -= s_1_pre

    s_intcpt = gbf._diversion_share_boundary_intcpt(
        s_1_pre,
        _delta_star,
        _r_val,
        recapture_form=recapture_form,
        aggregator=aggregator,
        weighting=weighting,
    )
    print(s_1_pre, _delta_star, _r_val, s_intcpt)
    print(type(s_intcpt))

    if weighting == "own-share":
        bdry_prtlarea = (
            _bdry_step_sz * (4 * s_2_oddsum + 2 * s_2_evnsum + _s_mid + s_2_pre) / 3
        )
        # Area under boundary
        bdry_area_total = 2 * (s_1_pre + bdry_prtlarea) - (
            mp.power(_s_mid, "2") + mp.power(s_1_pre, "2")
        )

    else:
        print([type(_f) for _f in (_s_mid, s_2_oddsum, s_2_evnsum, s_intcpt)])
        bdry_prtlarea = (
            _bdry_step_sz * (4 * s_2_oddsum + 2 * s_2_evnsum + _s_mid + s_intcpt) / 3
        )
        # Area under boundary
        bdry_area_total = 2 * bdry_prtlarea - mp.power(_s_mid, "2")

    bdry_points.append((mpf("0.0"), s_intcpt))
    # Points defining boundary to point-of-symmetry
    return GuidelinesBoundary(
        np.asarray(np.vstack((bdry_points[::-1], np.flip(bdry_points[1:], 1))), float),
        round(float(bdry_area_total), dps),
    )


def diversion_share_boundary_xact_avg_mp(
    _delta_star: float | MPFloat = 0.075,
    _r_val: float | MPFloat = DEFAULT_REC,
    /,
    *,
    recapture_form: Literal["inside-out", "fixed"] = "inside-out",
    dps: int = 5,
) -> GuidelinesBoundary:
    R"""
    Share combinations along the simple average diversion-ratio boundary.

    Parameters
    ----------
    _delta_star
        Diversion share, :math:`\overline{d} / \overline{r}` or :math:`\overline{g} / (m^* \cdot \overline{r})`.
    _r_val
        Recapture rate.
    recapture_form
        Whether recapture rate is share-proportional ("inside-out") or has fixed
        value for both merging firms ("fixed").
    dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary, area under boundary.

    Notes
    -----
    An analytical expression for the exact average boundary is derived
    and plotted from the y-intercept to the ray of symmetry as follows::

        from sympy import latex, plot as symplot, solve, symbols

        s_1, s_2 = symbols("s_1 s_2")

        g_val, r_val, m_val = 0.06, 0.80, 0.30
        d_hat = g_val / (r_val * m_val)

        # recapture_form = "inside-out"
        sag = solve(
            (s_2 / (1 - s_1))
            + (s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1)))
            - 2 * d_hat,
            s_2,
        )[0]
        symplot(sag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)

        # recapture_form = "fixed"
        sag = solve(
            (s_2 / (1 - s_1)) + (s_1 / (1 - s_2)) - 2 * d_hat, s_2
        )[0]
        symplot(sag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)

    """
    _delta_star, _r_val = (mpf(_v) for _v in (f"{_delta_star}", f"{_r_val}"))
    _s_mid = _delta_star / (1 + _delta_star)
    _bdry_step_sz = 10**-dps
    _bdry_start = np.array([(_s_mid, _s_mid)])
    _s_1 = np.array(mp.arange(_s_mid - _bdry_step_sz, 0, -_bdry_step_sz))
    if recapture_form == "inside-out":
        s_intcpt = mp.fdiv(
            mp.fsub(
                2 * _delta_star * _r_val + 1, mp.fabs(2 * _delta_star * _r_val - 1)
            ),
            2 * mpf(f"{_r_val}"),
        )
        nr_t1 = 1 + 2 * _delta_star * _r_val * (1 - _s_1) - _s_1 * (1 - _r_val)

        nr_sqrt_mdr = 4 * _delta_star * _r_val
        nr_sqrt_mdr2 = nr_sqrt_mdr * _r_val
        nr_sqrt_md2r2 = nr_sqrt_mdr2 * _delta_star

        nr_sqrt_t1 = nr_sqrt_md2r2 * (_s_1**2 - 2 * _s_1 + 1)
        nr_sqrt_t2 = nr_sqrt_mdr2 * _s_1 * (_s_1 - 1)
        nr_sqrt_t3 = nr_sqrt_mdr * (2 * _s_1 - _s_1**2 - 1)
        nr_sqrt_t4 = (_s_1**2) * (_r_val**2 - 6 * _r_val + 1)
        nr_sqrt_t5 = _s_1 * (6 * _r_val - 2) + 1

        nr_t2_mdr = nr_sqrt_t1 + nr_sqrt_t2 + nr_sqrt_t3 + nr_sqrt_t4 + nr_sqrt_t5

        # Alternative grouping of terms in np.sqrt
        nr_sqrt_s1sq = (_s_1**2) * (
            nr_sqrt_md2r2 + nr_sqrt_mdr2 - nr_sqrt_mdr + _r_val**2 - 6 * _r_val + 1
        )
        nr_sqrt_s1 = _s_1 * (
            -2 * nr_sqrt_md2r2 - nr_sqrt_mdr2 + 2 * nr_sqrt_mdr + 6 * _r_val - 2
        )
        nr_sqrt_nos1 = nr_sqrt_md2r2 - nr_sqrt_mdr + 1

        nr_t2_s1 = nr_sqrt_s1sq + nr_sqrt_s1 + nr_sqrt_nos1

        if not np.isclose(
            np.einsum("i->", nr_t2_mdr.astype(float)),
            np.einsum("i->", nr_t2_s1.astype(float)),
            rtol=0,
            atol=0.5 * dps,
        ):
            raise RuntimeError(
                "Calculation of sq. root term in exact average GUPPI"
                f"with recapture spec, {f'"{recapture_form}"'} is incorrect."
            )

        s_2 = (nr_t1 - np.sqrt(nr_t2_s1)) / (2 * _r_val)

    else:
        s_intcpt = mp.fsub(_delta_star + 1 / 2, mp.fabs(_delta_star - 1 / 2))
        s_2 = (
            (1 / 2)
            + _delta_star
            - _delta_star * _s_1
            - np.sqrt(
                ((_delta_star**2) - 1) * (_s_1**2)
                + (-2 * (_delta_star**2) + _delta_star + 1) * _s_1
                + (_delta_star**2)
                - _delta_star
                + (1 / 4)
            )
        )

    bdry_inner = np.stack((_s_1, s_2), axis=1)
    bdry_end = np.array([(mpf("0.0"), s_intcpt)])

    bdry = np.vstack((
        bdry_end,
        bdry_inner[::-1],
        _bdry_start,
        np.flip(bdry_inner, 1),
        np.flip(bdry_end, 1),
    ))
    s_2 = np.concatenate((np.array([_s_mid]), s_2))

    bdry_ends = [0, -1]
    bdry_odds = np.array(range(1, len(s_2), 2), int)
    bdry_evns = np.array(range(2, len(s_2), 2), int)

    # Double the area under the half-curve, and subtract the double-counted bit.
    bdry_area_simpson = 2 * _bdry_step_sz * (
        (4 / 3) * np.sum(s_2.take(bdry_odds))
        + (2 / 3) * np.sum(s_2.take(bdry_evns))
        + (1 / 3) * np.sum(s_2.take(bdry_ends))
    ) - mp.power(_s_mid, 2)

    return GuidelinesBoundary(bdry, float(mp.nstr(bdry_area_simpson, dps)))


# diversion_share_boundary_wtd_avg_autoroot
# this function is about half as fast as the manual one! ... and a touch less precise
def _diversion_share_boundary_wtd_avg_autoroot(
    _delta_star: float | MPFloat = 0.075,
    _r_val: float | MPFloat = DEFAULT_REC,
    /,
    *,
    aggregator: Literal[
        "arithmetic mean", "geometric mean", "distance"
    ] = "arithmetic mean",
    weighting: Literal["own-share", "cross-product-share", None] = "own-share",
    recapture_form: Literal["inside-out", "fixed"] = "inside-out",
    dps: int = 5,
) -> GuidelinesBoundary:
    R"""
    Share combinations on the share-weighted average diversion share boundary.

    Parameters
    ----------
    _delta_star
        Diversion share, :math:`\overline{d} / \overline{r}` or :math:`\overline{g} / (m^* \cdot \overline{r})`.
    _r_val
        Recapture rate.
    aggregator
        Whether "arithmetic mean", "geometric mean", or "distance".
    weighting
        Whether "own-share" or "cross-product-share"  (or None for simple, unweighted average).
    recapture_form
        Whether recapture rate is share-proportional ("inside-out") or has fixed
        value for both merging firms ("fixed").
    dps
        Number of decimal places for rounding returned shares and area.

    Returns
    -------
        Array of share-pairs, area under boundary.

    Notes
    -----
    An analytical expression for the share-weighted arithmetic mean boundary
    is derived and plotted from y-intercept to the ray of symmetry as follows::

        from sympy import plot as symplot, solve, symbols

        s_1, s_2 = symbols("s_1 s_2", positive=True)

        g_val, r_val, m_val = 0.06, 0.80, 0.30
        delta_star = g_val / (r_val * m_val)

        # recapture_form == "inside-out"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1))
            - (s_1 + s_2) * delta_star,
            s_2,
        )[0]
        symplot(oswag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)

        cpswag = solve(
            s_2 * s_2 / (1 - s_1)
            + s_1 * s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1))
            - (s_1 + s_2) * delta_star,
            s_2,
        )[1]
        symplot(cpwag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)

        # recapture_form == "fixed"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - s_2)
            - (s_1 + s_2) * delta_star,
            s_2,
        )[0]
        symplot(oswag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)

        cpswag = solve(
            s_2 * s_2 / (1 - s_1)
            + s_1 * s_1 / (1 - s_2)
            - (s_1 + s_2) * delta_star,
            s_2,
        )[1]
        symplot(cpswag, (s_1, 0.0, d_hat / (1 + d_hat)), ylabel=s_2)


    """
    _delta_star, _r_val = (mpf(f"{_v}") for _v in (_delta_star, _r_val))
    _s_mid = mp.fdiv(_delta_star, 1 + _delta_star)

    # initial conditions
    bdry = [(_s_mid, _s_mid)]
    s_1_pre, s_2_pre = _s_mid, _s_mid
    s_2_oddval, s_2_oddsum, s_2_evnsum = True, 0.0, 0.0

    # parameters for iteration
    _step_size = mp.power(10, -dps)
    theta_ = _step_size * (10 if weighting == "cross-product-share" else 1)
    for s_1 in mp.arange(_s_mid - _step_size, 0, -_step_size):

        def delta_test(x: MPFloat) -> MPFloat:
            _de_1 = x / (1 - s_1)
            _de_2 = (
                s_1 / (1 - gbf.lerp(s_1, x, _r_val))
                if recapture_form == "inside-out"
                else s_1 / (1 - x)
            )
            _w = (
                mp.fdiv(s_1 if weighting == "cross-product-share" else x, s_1 + x)
                if weighting
                else 0.5
            )

            match aggregator:
                case "geometric mean":
                    delta_test = mp.expm1(
                        gbf.lerp(mp.log1p(_de_1), mp.log1p(_de_2), _w)
                    )
                case "distance":
                    delta_test = mp.sqrt(gbf.lerp(_de_1**2, _de_2**2, _w))
                case _:
                    delta_test = gbf.lerp(_de_1, _de_2, _w)

            return _delta_star - delta_test

        try:
            s_2 = mp.findroot(
                delta_test,
                x0=(s_2_pre * (1 - theta_), s_2_pre * (1 + theta_)),
                tol=mp.sqrt(_step_size),
                solver="ridder",
            )
        except (mp.ComplexResult, ValueError, ZeroDivisionError) as _e:
            print(s_1, s_2_pre)
            raise _e

        # Build-up boundary points
        bdry.append((s_1, s_2))

        # Build up area terms
        s_2_oddsum += s_2 if s_2_oddval else 0
        s_2_evnsum += s_2 if not s_2_oddval else 0
        s_2_oddval = not s_2_oddval

        # Hold share points
        s_2_pre = s_2
        s_1_pre = s_1

        if (s_1_pre + s_2_pre) > mpf("0.99875"):
            # Loss of accuracy at 3-9s and up
            break

    if s_2_oddval:
        s_2_evnsum -= s_2_pre
    else:
        s_2_oddsum -= s_1_pre

    _s_intcpt = gbf._diversion_share_boundary_intcpt(
        s_2_pre,
        _delta_star,
        _r_val,
        recapture_form=recapture_form,
        aggregator=aggregator,
        weighting=weighting,
    )

    if weighting == "own-share":
        gbd_prtlarea = (
            _step_size * (4 * s_2_oddsum + 2 * s_2_evnsum + _s_mid + s_2_pre) / 3
        )
        # Area under boundary
        bdry_area_total = float(
            2 * (s_1_pre + gbd_prtlarea)
            - (mp.power(_s_mid, "2") + mp.power(s_1_pre, "2"))
        )

    else:
        gbd_prtlarea = (
            _step_size * (4 * s_2_oddsum + 2 * s_2_evnsum + _s_mid + _s_intcpt) / 3
        )
        # Area under boundary
        bdry_area_total = float(2 * gbd_prtlarea - mp.power(_s_mid, "2"))

    bdry.append((mpf("0.0"), _s_intcpt))
    bdry_array = np.array(bdry, float)

    # Points defining boundary to point-of-symmetry
    return GuidelinesBoundary(
        np.vstack((bdry_array[::-1], bdry_array[1:, ::-1]), dtype=float),
        round(float(bdry_area_total), dps),
    )
