"""Helper functions for defining and analyzing boundaries for Guidelines standards."""

import decimal
from collections.abc import Callable
from types import ModuleType
from typing import Literal
from typing import TypedDict

import matplotlib as mpl
import matplotlib.axes as mpa
import matplotlib.offsetbox as mof
import matplotlib.patches as mpp
import matplotlib.pyplot as plt
import matplotlib.ticker as mpt
import numpy as np
from matplotlib import figure
from mpmath import mp
from mpmath import mpf

from .. import DEFAULT_REC
from .. import PKG_NAME
from .. import VERSION
from .. import ArrayDouble
from .. import MPFloat
from . import GuidelinesBoundary

__version__ = VERSION

mp.dps = 32
mp.trap_complex = True


class DiversionShareBoundaryKeywords(TypedDict, total=False):
    """Keyword arguments for functions generating share ratio boundaries."""

    recapture_form: Literal["fixed", "inside-out", "outside-in"]
    dps: int
    aggregator: Literal["arithmetic mean", "geometric mean", "distance", "max", "min"]
    weighting: Literal["own-share", "cross-product-share"] | None


def dh_area(_delta_bound: float | MPFloat = 0.01, /, *, dps: int = 9) -> float:
    R"""
    Area under the ΔHHI boundary.

    When the given ΔHHI bound matches a Guidelines standard,
    the area under the boundary is half the intrinsic clearance rate
    for the ΔHHI safeharbor.

    Parameters
    ----------
    _delta_bound
        Change in concentration.
    dps
        Specified precision in decimal places.

    Returns
    -------
        Area under ΔHHI boundary.


    Notes
    -----
    To derive the knots, :math:`(s^0_1, s^1_1), (s^1_1, s^0_1)`
    of the ΔHHI boundary, i.e., the points where it intersects
    the merger-to-monopoly boundary, solve

    .. math::

        2 s1 s_2 &= ΔHHI \\
        s_1 + s_2 &= 1

    """
    _delta_bound = mpf(f"{_delta_bound}")
    _s_naught = (1 - mp.sqrt(1 - 2 * _delta_bound)) / 2

    return round(
        float(
            _s_naught + (_delta_bound / 2) * (mp.ln(1 - _s_naught) - mp.ln(_s_naught))
        ),
        dps,
    )


def hhi_delta_boundary(
    _delta_bound: float | decimal.Decimal | MPFloat = 0.01, /, *, dps: int = 5
) -> GuidelinesBoundary:
    """
    Generate the list of share combination on the ΔHHI boundary.

    Parameters
    ----------
    _delta_bound:
        Merging-firms' ΔHHI bound.
    dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _delta_bound = mpf(f"{_delta_bound}")
    _s_naught = 1 / 2 * (1 - mp.sqrt(1 - 2 * _delta_bound))
    _s_mid = mp.sqrt(_delta_bound / 2)

    _step_size = mp.power(10, -6)
    _s_1 = np.array(mp.arange(_s_mid, _s_naught - mp.eps, -_step_size))

    # Boundary points
    half_bdry = np.vstack((
        np.stack((_s_1, _delta_bound / (2 * _s_1)), axis=1),
        np.array([(mpf("0.0"), mpf("1.0"))]),
    ))
    bdry = np.vstack((half_bdry[::-1], half_bdry[1:, ::-1]))

    return GuidelinesBoundary(bdry, dh_area(_delta_bound, dps=dps))


def hhi_pre_contrib_boundary(
    _hhi_bound: float | decimal.Decimal | MPFloat = 0.03125, /, *, dps: int = 5
) -> GuidelinesBoundary:
    """
    Share combinations on the premerger HHI contribution boundary.

    Parameters
    ----------
    _hhi_bound:
        Merging-firms' pre-merger HHI contribution bound.
    dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _hhi_bound = mpf(f"{_hhi_bound}")
    _s_mid = mp.sqrt(_hhi_bound / 2)

    step_size = mp.power(10, -dps)
    # Range-limit is 0 less a step, which is -1 * step-size
    s_1 = np.array(mp.arange(_s_mid, -step_size, -step_size))
    s_2 = np.sqrt(_hhi_bound - s_1**2)
    half_bdry = np.stack((s_1, s_2), axis=1)

    return GuidelinesBoundary(
        np.vstack((half_bdry[::-1], half_bdry[1:, ::-1])),
        round(float(mp.pi * _hhi_bound / 4), dps),
    )


def combined_share_boundary(
    _s_intcpt: float | decimal.Decimal | MPFloat = 0.0625, /, *, dps: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the merging-firms' combined share boundary.

    Assumes symmetric merging-firm margins. The combined-share is
    congruent to the post-merger HHI contribution boundary, as the
    post-merger HHI bound is the square of the combined-share bound.

    Parameters
    ----------
    _s_intcpt:
        Merging-firms' combined share.
    dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _s_intcpt = mpf(f"{_s_intcpt}")
    _s_mid = _s_intcpt / 2

    _s1 = np.array([0, _s_mid, _s_intcpt], float)
    return GuidelinesBoundary(
        list(zip(_s1, _s1[::-1], strict=True)), round(float(_s_intcpt * _s_mid), dps)
    )


def hhi_post_contrib_boundary(
    _hhi_bound: float | decimal.Decimal | MPFloat = 0.800, /, *, dps: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the postmerger HHI contribution boundary.

    The post-merger HHI contribution boundary is identical to the
    combined-share boundary.

    Parameters
    ----------
    _hhi_bound:
        Merging-firms' pre-merger HHI contribution bound.
    dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    return combined_share_boundary(
        _hhi_bound.sqrt()
        if isinstance(_hhi_bound, decimal.Decimal | mpf)
        else np.sqrt(_hhi_bound),
        dps=dps,
    )


# hand-rolled root finding
def diversion_share_boundary_wtd_avg(
    _delta_star: float = 0.075,
    _r_val: float = DEFAULT_REC,
    /,
    *,
    aggregator: Literal[
        "arithmetic mean", "geometric mean", "distance"
    ] = "arithmetic mean",
    weighting: Literal["own-share", "cross-product-share"] | None = "own-share",
    recapture_form: Literal["fixed", "inside-out", "outside-in"] = "inside-out",
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
    if recapture_form == "outside-in":
        raise NotImplementedError(
            f"Boundary function not implemented for, recapture_form={recapture_form!r}. "
            "Retry with 'inside-out' or 'fixed'."
        )

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
        # The wtd. avg. GUPPI is not always convex to the origin, so we
        #   increment s_2 after each iteration in which our algorithm
        #   finds (s1, s2) on the boundary
        s_2 = s_2_pre * (1 + theta_)

        while True:
            de_1 = s_2 / (1 - s_1)
            de_2 = (
                s_1 / (1 - lerp(s_1, s_2, _r_val))
                if recapture_form == "inside-out"
                else s_1 / (1 - s_2)
            )

            r_ = (
                mp.fdiv(s_1 if weighting == "cross-product-share" else s_2, s_1 + s_2)
                if weighting
                else 0.5
            )

            match aggregator:
                case "geometric mean":
                    delta_test = mp.expm1(lerp(mp.log1p(de_1), mp.log1p(de_2), r_))
                case "distance":
                    delta_test = mp.sqrt(lerp(de_1**2, de_2**2, r_))
                case _:
                    delta_test = lerp(de_1, de_2, r_)

            test_flag, incr_decr = (
                (delta_test > _delta_star, -1)
                if weighting == "cross-product-share"
                else (delta_test < _delta_star, 1)
            )

            if test_flag:
                s_2 += incr_decr * _step_size
            else:
                break

        # Build-up boundary points
        bdry.append((s_1, s_2))

        # Build up area terms
        s_2_oddsum += s_2 if s_2_oddval else 0
        s_2_evnsum += s_2 if not s_2_oddval else 0
        s_2_oddval = not s_2_oddval

        # Hold share points
        s_2_pre = s_2
        s_1_pre = s_1

        if (s_1 + s_2) > mpf("0.99875"):
            # Loss of accuracy at 3-9s and up
            break

    if s_2_oddval:
        s_2_evnsum -= s_2_pre
    else:
        s_2_oddsum -= s_1_pre

    _s_intcpt = _diversion_share_boundary_intcpt(
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
    bdry_array = ArrayDouble(bdry)

    # Points defining boundary to point-of-symmetry
    return GuidelinesBoundary(
        np.vstack((bdry_array[::-1, :], bdry_array[1:, ::-1])),
        round(float(bdry_area_total), dps),
    )


def diversion_share_boundary_xact_avg(
    _delta_star: float = 0.075,
    _r_val: float = DEFAULT_REC,
    /,
    *,
    recapture_form: Literal["inside-out", "fixed", "outside-in"] = "inside-out",
    dps: int = 5,
) -> GuidelinesBoundary:
    R"""
    Share combinations for the exact average diversion share boundary.

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
    if recapture_form == "outside-in":
        raise NotImplementedError(
            f"Boundary function not implemented for, recapture_form={recapture_form!r}. "
            "Retry with 'inside-out' or 'fixed'."
        )

    _delta_star, _r_val = (mpf(f"{_v}") for _v in (_delta_star, _r_val))
    _s_mid = _delta_star / (1 + _delta_star)
    _step_size = mp.power("10", f"-{dps}")

    _bdry_start = np.array([(_s_mid, _s_mid)])
    _s_1 = np.asarray(mp.arange(_s_mid - _step_size, 0, -_step_size))
    if recapture_form == "inside-out":
        _s_intcpt = (
            2 * _delta_star * _r_val + 1 - np.abs(2 * _delta_star * _r_val - 1)
        ) / (2 * _r_val)
        nr_t1 = 1 + 2 * _delta_star * _r_val * (1 - _s_1) - _s_1 * (1 - _r_val)

        nr_sqrt_mdr: float = 4 * _delta_star * _r_val
        nr_sqrt_mdr2: float = nr_sqrt_mdr * _r_val
        nr_sqrt_md2r2: float = nr_sqrt_mdr2 * _delta_star

        nr_sqrt_t1 = nr_sqrt_md2r2 * (_s_1**2 - 2 * _s_1 + 1)
        nr_sqrt_t2 = nr_sqrt_mdr2 * _s_1 * (_s_1 - 1)
        nr_sqrt_t3 = nr_sqrt_mdr * (2 * _s_1 - _s_1**2 - 1)
        nr_sqrt_t4 = (_s_1**2) * (_r_val**2 - 6 * _r_val + 1)
        nr_sqrt_t5 = _s_1 * (6 * _r_val - 2) + 1

        nr_t2_mdr = nr_sqrt_t1 + nr_sqrt_t2 + nr_sqrt_t3 + nr_sqrt_t4 + nr_sqrt_t5

        # Alternative grouping of terms in np.sqrt
        nr_sqrt_nos1: float = nr_sqrt_md2r2 - nr_sqrt_mdr + 1
        nr_sqrt_s1 = _s_1 * (
            -2 * nr_sqrt_md2r2 - nr_sqrt_mdr2 + 2 * nr_sqrt_mdr + 6 * _r_val - 2
        )
        nr_sqrt_s1sq = (_s_1**2) * (
            nr_sqrt_md2r2 + nr_sqrt_mdr2 - nr_sqrt_mdr + _r_val**2 - 6 * _r_val + 1
        )
        nr_t2_s1 = nr_sqrt_s1sq + nr_sqrt_s1 + nr_sqrt_nos1

        if not np.isclose(
            float(np.einsum("i->", nr_t2_mdr)),
            float(np.einsum("i->", nr_t2_s1)),
            rtol=0,
            atol=0.5 * dps,
        ):
            raise RuntimeError(
                "Calculation of sq. root term in exact average GUPPI"
                f"with recapture spec, {f'"{recapture_form}"'} is incorrect."
            )

        s_2 = (nr_t1 - np.sqrt(nr_t2_s1)) / (2 * _r_val)

    else:
        _s_intcpt = _delta_star + 1 / 2 - np.abs(_delta_star - 1 / 2)
        s_2 = (
            0.5
            + _delta_star
            - _delta_star * _s_1
            - (
                (
                    ((_delta_star**2) - 1) * (_s_1**2)
                    + (-2 * (_delta_star**2) + _delta_star + 1) * _s_1
                    + (_delta_star**2)
                    - _delta_star
                    + (1 / 4)
                )
                ** 0.5
            )
        )

    bdry_inner = np.stack((_s_1, s_2), axis=1)
    bdry_end = np.array([(mpf("0.0"), _s_intcpt)])

    bdry = np.vstack((
        bdry_end,
        bdry_inner[::-1],
        _bdry_start,
        bdry_inner[:, ::-1],
        bdry_end[:, ::-1],
    ))
    s_2 = np.concatenate(([_s_mid], s_2))

    bdry_idxs_ends = [0, -1]
    bdry_idxs_odds = np.arange(1, len(s_2), 2)
    bdry_idxs_evns = np.arange(2, len(s_2), 2)

    # Double the area under the curve, and subtract the double counted bit.
    bdry_area_simpson = 2 * _step_size * (
        (4 / 3) * np.sum(s_2.take(bdry_idxs_odds))
        + (2 / 3) * np.sum(s_2.take(bdry_idxs_evns))
        + (1 / 3) * np.sum(s_2.take(bdry_idxs_ends))
    ) - (_s_mid * _s_mid)

    return GuidelinesBoundary(bdry, round(float(bdry_area_simpson), dps))


def diversion_share_boundary_min(
    _delta_star: float = 0.075,
    _r_val: float = DEFAULT_REC,
    /,
    *,
    recapture_form: Literal["inside-out", "fixed"] = "inside-out",
    dps: int = 10,
) -> GuidelinesBoundary:
    R"""
    Share combinations on the minimum diversion-ratio/share-ratio boundary.

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
        Array of share-pairs, area under boundary.

    Notes
    -----
    With symmetric merging-firm margins, the maximum GUPPI boundary is
    defined by the diversion ratio from the smaller merging-firm to the
    larger one, and is hence unaffected by the method of estimating the
    diversion ratio for the larger firm.

    """
    _delta_star, _r_val = (mpf(f"{_v}") for _v in (_delta_star, _r_val))
    _s_intcpt = mpf("1.00")
    _s_mid = _delta_star / (1 + _delta_star)

    if recapture_form == "inside-out":
        # ## Plot envelope of GUPPI boundaries with rk_ = r_bar if sk_ = min(s_1, s_2)
        # ## See (si_, sj_) in equation~(44), or thereabouts, in paper
        smin_nr = _delta_star * (1 - _r_val)
        smax_nr = 1 - _delta_star * _r_val
        dr = smin_nr + smax_nr
        s_1 = np.array([0, smin_nr / dr, _s_mid, smax_nr / dr, _s_intcpt])

        bdry_area = _s_mid + s_1[1] * (1 - 2 * _s_mid)
    else:
        s_1, bdry_area = np.array([0, _s_mid, _s_intcpt]), _s_mid

    return GuidelinesBoundary(
        list(zip(s_1, s_1[::-1], strict=True)), round(float(bdry_area), dps)
    )


def diversion_share_boundary_max(
    _delta_star: float = 0.075, _: float = DEFAULT_REC, /, *, dps: int = 10
) -> GuidelinesBoundary:
    R"""
    Share combinations on the minimum diversion-ratio/share-ratio boundary.

    Parameters
    ----------
    _delta_star
        Diversion share, :math:`\overline{d} / \overline{r}` or :math:`\overline{g} / (m^* \cdot \overline{r})`.
    _
        Placeholder for recapture rate included for consistency with other
        share-ratio boundary functions.
    dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _delta_star = mpf(f"{_delta_star}")
    _s_intcpt = _delta_star
    _s_mid = _delta_star / (1 + _delta_star)
    _s1_pts = (0, _s_mid, _s_intcpt)

    return GuidelinesBoundary(
        list(zip(_s1_pts, _s1_pts[::-1], strict=True)),
        round(float(_s_intcpt * _s_mid), dps),  # simplified calculation
    )


def _diversion_share_boundary_intcpt(
    s_2_pre: float,
    _delta_star: MPFloat,
    _r_val: MPFloat,
    /,
    *,
    recapture_form: Literal["inside-out", "fixed"],
    aggregator: Literal["arithmetic mean", "geometric mean", "distance"],
    weighting: Literal["cross-product-share", "own-share"] | None,
) -> float:
    match weighting:
        case "cross-product-share":
            _s_intcpt: float = _delta_star
        case "own-share":
            _s_intcpt = mpf("1.0")
        case None if aggregator == "distance":
            _s_intcpt = _delta_star * mp.sqrt("2")
        case None if aggregator == "arithmetic mean" and recapture_form == "inside-out":
            _s_intcpt = mp.fdiv(
                mp.fsub(
                    2 * _delta_star * _r_val + 1, mp.fabs(2 * _delta_star * _r_val - 1)
                ),
                2 * mpf(f"{_r_val}"),
            )
        case None if aggregator == "arithmetic mean" and recapture_form == "fixed":
            _s_intcpt = mp.fsub(_delta_star + 1 / 2, mp.fabs(_delta_star - 1 / 2))
        case _:
            _s_intcpt = s_2_pre

    return _s_intcpt


def lerp[LerpT: (float, MPFloat, ArrayDouble)](
    _x1: LerpT, _x2: LerpT, _r: float | MPFloat = 0.25, /
) -> LerpT:
    R"""
    From the function of the same name in the C++ standard [#]_. Also, [#]_.

    Constructs the weighted average, :math:`w_1 x_1 + w_2 x_2`, where
    :math:`w_1 = 1 - r` and :math:`w_2 = r`.

    Parameters
    ----------
    _x1, _x2
        Interpolation bounds :math:`x_1, x_2`.
    _r
        Interpolation weight :math:`r` assigned to :math:`x_2`

    Returns
    -------
        The linear interpolation, or weighted average,
        :math:`x_1 + r \cdot (x_2 - x_1) \equiv (1 - r) \cdot x_1 + r \cdot x_2`.

    Raises
    ------
    ValueError
        If the interpolation weight is not in the interval, :math:`[0, 1]`.

    References
    ----------

    .. [#] C++ Reference, https://en.cppreference.com/w/cpp/numeric/lerp

    .. [#] Harris, Mark (2015). "GPU Pro Tip: Lerp faster in C++". nvidia.com (Jun 10, 2015). Online at:
        https://developer.nvidia.com/blog/lerp-faster-cuda/


    """
    match _r:
        case 0:
            return _x1
        case 1:
            return _x2
        case _:
            if not 0 < _r < 1:
                raise ValueError("Specified interpolation weight must lie in [0, 1].")
            return _x1 + _r * (_x2 - _x1)  # type: ignore[return-value]


def round_cust(
    _num: float | decimal.Decimal | MPFloat = 0.060215,
    /,
    *,
    frac: float = 0.005,
    rounding_mode: str = "ROUND_HALF_UP",
) -> float:
    """
    Round to given fraction; the nearest 0.5% by default.

    Parameters
    ----------
    _num
        Number to be rounded.
    frac
        Fraction to be rounded to.
    rounding_mode
        Rounding mode, as defined in the :code:`decimal` package.

    Returns
    -------
        The given number, rounded as specified.

    Raises
    ------
    ValueError
        If rounding mode is not defined in the :code:`decimal` package.

    Notes
    -----
    Integer-round the quotient, :code:`(_num / frac)` using the specified
    rounding mode. Return the product of the rounded quotient times
    the specified precision, :code:`frac`.

    """
    if rounding_mode not in {
        decimal.ROUND_05UP,
        decimal.ROUND_CEILING,
        decimal.ROUND_DOWN,
        decimal.ROUND_FLOOR,
        decimal.ROUND_HALF_DOWN,
        decimal.ROUND_HALF_EVEN,
        decimal.ROUND_HALF_UP,
        decimal.ROUND_UP,
    }:
        raise ValueError(
            f'Value, "{rounding_mode}" is invalid for rounding_mode. '
            "For valid rounding modes, see documentation for "
            'the built-in module, "decimal".'
        )

    _n, _f, _e = (decimal.Decimal(f"{_g}") for _g in [_num, frac, 1])

    return float(_f * (_n / _f).quantize(_e, rounding=rounding_mode))


def boundary_plot(
    _plt: ModuleType,
    /,
    *,
    mktshare_plot_flag: bool = True,
    mktshare_axes_flag: bool = True,
    backend: str | None = "pgf",
) -> tuple[figure.Figure, Callable[[mpa.Axes, bool, bool], None]]:
    """Set up basic figure and axes for plots of safe harbor boundaries.

    See, https://matplotlib.org/stable/tutorials/text/pgf.html
    """
    if backend == "pgf":
        mpl.use("pgf")

        mpl.rcParams.update({
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                R"\usepackage{fontspec}",
                R"\usepackage{luacode}",
                R"\begin{luacode}",
                R"local function embedfull(tfmdata)",
                R'    tfmdata.embedding = "full"',
                R"end",
                R"",
                R"luatexbase.add_to_callback("
                R'  "luaotfload.patch_font", embedfull, "embedfull"'
                R")",
                R"\end{luacode}",
                R"\defaultfontfeatures[\rmfamily]{",
                R"  Ligatures={TeX, Common},",
                R"  Numbers={Proportional, Lining},",
                R"  }",
                R"\defaultfontfeatures[\sffamily, \dvsfamily]{",
                R"  Ligatures={TeX, Common},",
                R"  Numbers={Monospaced, Lining},",
                R"  LetterSpace=0.50,",
                R"  }",
                R"\setmainfont{STIX Two Text}",
                R"\setsansfont{Fira Sans Light}",
                R"\setmonofont[Scale=MatchLowercase,]{Fira Mono}",
                R"\newfontfamily\dvsfamily{DejaVu Sans}",
                R"\usepackage{mathtools}",
                R"\usepackage{unicode-math}",
                R"\setmathfont{STIX Two Math}[math-style=ISO,bold-style=ISO]",
                R"\setmathfont{STIX Two Math}[math-style=ISO,range={scr,bfscr},StylisticSet=01]",
                R"\usepackage[",
                R"  activate={true, nocompatibility},",
                R"  tracking=true,",
                R"  ]{microtype}",
            ]),
        })
    else:
        if backend:
            mpl.use(backend)
        mpl.rcParams.update({
            "text.usetex": False,
            "pgf.rcfonts": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "sans-serif"],
            "font.monospace": ["DejaVu Mono", "monospace"],
            "font.serif": ["stix", "serif"],
            "mathtext.fontset": "stix",
        })

    # Initialize a canvas with a single figure (set of axes)
    fig = plt.figure(figsize=(5, 5), dpi=600)
    ax = fig.add_subplot()
    # Set the width of axis grid lines, and tick marks:
    # both axes, both major and minor ticks
    # Frame, grid, and face color
    for _spos0 in "left", "bottom":
        ax.spines[_spos0].set_linewidth(0.5)
        ax.spines[_spos0].set_zorder(5)
    for _spos1 in "top", "right":
        ax.spines[_spos1].set_linewidth(0.0)
        ax.spines[_spos1].set_zorder(0)
        ax.spines[_spos1].set_visible(False)
    ax.set_facecolor("0.875")

    ax.grid(linewidth=0.5, linestyle=":", color="grey", zorder=1)
    ax.tick_params(axis="both", which="both", width=0.5)

    # Tick marks skip, size, and rotation
    # x-axis
    for _t in ax.get_xticklabels():
        _t.update({"fontsize": 6, "rotation": 45, "ha": "right"})
    # y-axis
    for _t in ax.get_yticklabels():
        _t.update({"fontsize": 6, "rotation": 0, "ha": "right"})

    def _set_axis_def(
        ax0: mpa.Axes,
        /,
        *,
        mktshare_plot_flag: bool = False,
        mktshare_axes_flag: bool = False,
    ) -> None:
        if mktshare_plot_flag:
            # Axis scale
            ax0.set_xlim(0, 1)
            ax0.set_ylim(0, 1)
            ax0.set_aspect(1.0)

            # Plot the ray of symmetry
            ax0.plot(
                [0, 1], [0, 1], linewidth=0.5, linestyle=":", color="grey", zorder=1
            )

            # Truncate the axis frame to a triangle bounded by the other diagonal:
            ax0.plot(
                [0, 1], [1, 0], linestyle="-", linewidth=0.5, color="black", zorder=1
            )
            ax0.add_patch(
                mpp.Rectangle(
                    xy=(1.0025, 0.00),
                    width=1.1 * mp.sqrt(2),
                    height=1.1 * mp.sqrt(2),
                    angle=45,
                    color="white",
                    edgecolor=None,
                    fill=True,
                    clip_on=True,
                    zorder=5,
                )
            )

            # Axis Tick-mark locations
            # One can supply an argument to mpt.AutoMinorLocator to
            # specify a fixed number of minor intervals per major interval, e.g.:
            # minorLocator = mpt.AutoMinorLocator(2)
            # would lead to a single minor tick between major ticks.
            for axs in ax0.xaxis, ax0.yaxis:
                axs.set_major_locator(mpt.MultipleLocator(0.05))
                axs.set_minor_locator(mpt.AutoMinorLocator(5))
                # It"s always x when specifying the format
                axs.set_major_formatter(mpt.StrMethodFormatter("{x:>3.0%}"))

            # Hide every other tick-label
            for axl in ax0.get_xticklabels(), ax0.get_yticklabels():
                for _t in axl[::2]:
                    _t.set_visible(False)

            # package version badge
            # https://futurile.net/2016/03/14/partial-colouring-text-in-matplotlib-with-latex/
            badge_fmt_str = R"{{\dvsfamily {}}}" if backend == "pgf" else "{}"
            badge_txt_list = [
                badge_fmt_str.format(_s) for _s in [f"{PKG_NAME}", f"v{VERSION}"]
            ]

            badge_fmt_list = [
                _bf0 := {"color": "#fff", "backgroundcolor": "#555", "size": 2},
                _bf0 | {"backgroundcolor": "#007ec6"},
            ]

            badge_box = mof.HPacker(
                sep=2.6,
                children=[
                    mof.TextArea(_t, textprops=_f)
                    for _t, _f in zip(badge_txt_list, badge_fmt_list, strict=True)
                ],
            )

            ax0.add_artist(
                mof.AnnotationBbox(
                    badge_box,
                    xy=(0.5, 0.5),
                    xybox=(-0.05, -0.12),
                    xycoords="data",
                    boxcoords="data",
                    frameon=False,
                    pad=0,
                )
            )

            # Axis labels
            if mktshare_axes_flag:
                # x-axis
                ax0.set_xlabel("Firm 1 Market Share, $s_1$", fontsize=10)
                ax0.xaxis.set_label_coords(0.75, -0.1)
                # y-axis
                ax0.set_ylabel("Firm 2 Market Share, $s_2$", fontsize=10)
                ax0.yaxis.set_label_coords(-0.1, 0.75)

    _set_axis_def(
        ax, mktshare_plot_flag=mktshare_plot_flag, mktshare_axes_flag=mktshare_axes_flag
    )

    return fig, _set_axis_def  # type: ignore[return-value]
