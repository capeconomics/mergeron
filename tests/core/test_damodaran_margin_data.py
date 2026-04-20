"""Test that the generation of empirical margin data  is replicable.

We check here that parallel generation replicates results from
single-threaded iteration, provided that seed/rng is spawned in the same
pattern and order, which means that the process is replicable and also that
there are no errors from parallelization.
"""

from collections.abc import Generator
from math import ceil

import numpy as np
import pytest
from joblib import Parallel
from joblib import delayed
from numpy.random import SeedSequence

from mergeron import ArrayDouble
from mergeron.core import empirical_margin_distribution as dmd
from mergeron.core.pseudorandom_numbers import PPN_S
from mergeron.core.pseudorandom_numbers import prng
from mergeron.gen import SUBSAMPLE_SIZE
from mergeron.gen import PCMDistribution
from mergeron.gen.data_generation_functions import _margin_resampler
from mergeron.gen.data_generation_functions import (
    _margin_resampler_multimodal_multithreaded,
)
from mergeron.gen.data_generation_functions import _simple_resampler

NTHREADS = 8
QTILES = [
    *(_r := [0, *(_s := [0.005, 0.01, 0.025]), *[_t * 10 for _t in _s]]),
    0.5,
    *[1 - _u for _u in _r[::-1]],
]
SAMPLE_SIZE = 10**8
NCOLS = 2

_margin_data_raw = dmd.margin_data_builder(dmd.margin_data_getter())
_margin_data_vector = np.repeat(
    _margin_data_raw.average_gross_margins,
    (_margin_data_raw.firm_counts * 10**3).astype(int),
)
_margin_data_bandwidth = _margin_data_raw.bandwidth


@pytest.fixture(scope="module")
def sample_0() -> Generator[np.ndarray]:
    """Create test sample of empirical margin data.

    We presently test that the test sample has no infeasible values
    and no extreme values. We also test that parallel generation gives
    the same set of margins as single-threaded iteration.
    """
    _mdv, _mdc, _mdb = (
        _margin_data_vector,
        len(_margin_data_vector),
        _margin_data_bandwidth,
    )

    _iter_count = ceil(SAMPLE_SIZE / SUBSAMPLE_SIZE)
    _sseq = SeedSequence(entropy=PPN_S, pool_size=8)
    _rng1, _rng2 = prng(_sseq).spawn(2)
    _s0 = np.vstack([
        (
            _mdv[_r1i.integers(_mdc, size=(SUBSAMPLE_SIZE, 1))]
            + _mdb * _r2i.standard_normal(size=(SUBSAMPLE_SIZE, 1))
        )
        for _r1i, _r2i in zip(
            _rng1.spawn(_iter_count), _rng2.spawn(_iter_count), strict=True
        )
    ])
    yield _s0
    del _s0


def test_data_generation(sample_0) -> None:
    """Test that data generation falls within the feasible range."""
    sample_1 = sample_0[(sample_0 > 0.0) & (sample_0 < 1)][:, None]
    if len(sample_1) != SAMPLE_SIZE:
        print(
            len(sample_0),
            len(sample_1),
            _margin_data_raw.bandwidth,
            sample_0.min(),
            sample_0.max(),
        )
        print(repr(sample_0[(sample_0 <= 0.0) | (sample_0 >= 1)]))
        raise ValueError("Generated data are not within the feasible range.")


def test_parallel_data_generation_1(sample_0) -> None:
    """Test that parallel data generation is replicable.

    Joblib may not return from each thread in the order called. So,
    rather than test that the generated array is the same as the test
    array, we test that various quantiles are the same in both arrays.
    """
    s0l = len(sample_0)
    s0q = np.quantile(sample_0, q=QTILES)

    # repeat with multithreading
    _iter_count = max(NTHREADS, ceil(SAMPLE_SIZE / SUBSAMPLE_SIZE))
    _sseq = SeedSequence(entropy=PPN_S, pool_size=8)
    _sseq1, _sseq2 = _sseq.spawn(2)

    _ssz = ceil(SAMPLE_SIZE / _iter_count)
    _trunc_flag = _iter_count * _ssz > SAMPLE_SIZE
    _trunc_size = SAMPLE_SIZE % _ssz if _trunc_flag else _ssz

    sample_0p = np.vstack(
        Parallel(backend="threading", n_jobs=min(NTHREADS, _iter_count))(
            delayed(_simple_resampler)(
                _margin_data_vector,
                _margin_data_bandwidth,
                (
                    _trunc_size if (1 + _sseq1i.spawn_key[-1]) == _iter_count else _ssz,
                    1,
                ),
                prng(_sseq1i),
                prng(_sseq2i),
            )
            for _sseq1i, _sseq2i in zip(
                _sseq1.spawn(_iter_count), _sseq2.spawn(_iter_count), strict=True
            )
        )
    )

    s0pl = len(sample_0p)
    s0pq = np.quantile(sample_0p, q=QTILES)
    del sample_0p
    print(s0l, s0pl)
    print(np.hstack((s0q, s0pq)))
    if (s0pl != s0l) or not np.array_equal(s0pq, s0q):
        raise ValueError("Parallelized data does not match one-shot data generation.")


def test_parallel_data_generation_2(sample_0) -> None:
    """Test that parallel data generation is replicable.

    Joblib may not return from each thread in the order called. So,
    rather than test that the generated array is the same as the test
    array, we test that various quantiles are the same in both arrays.
    """
    s0l = len(sample_0)
    s0q = np.quantile(sample_0, q=QTILES)

    _sseq = SeedSequence(entropy=PPN_S, pool_size=8)
    _ssq1, _ssq2 = _sseq.spawn(2)

    # repeat with built-in multithreading
    sample_1p = _margin_resampler_multimodal_multithreaded(
        _margin_data_raw, (SAMPLE_SIZE, 1), NTHREADS, _ssq1, _ssq2
    )

    s1pl = len(sample_1p)
    s1pq = np.quantile(sample_1p, q=QTILES)
    del sample_1p
    print(s0l, s1pl)
    print(np.hstack((s0q, s1pq)))
    if (s1pl != s0l) or not np.array_equal(s1pq, s0q):
        raise ValueError("Margin vector sampler does not match hand-rolled sample.")


def test_parallel_data_generation_3() -> None:
    """Test that data generation against mean in source data."""
    _sseq = SeedSequence(entropy=PPN_S, pool_size=8)
    _ssq1, _ssq2 = _sseq.spawn(2)

    sample_1p = _margin_resampler_multimodal_multithreaded(
        _margin_data_raw, (SAMPLE_SIZE, NCOLS), NTHREADS, _ssq1, _ssq2
    ).view(ArrayDouble)

    if (
        len(sample_1p) != SAMPLE_SIZE
        or not np.isclose(
            sample_1p.mean(),
            _margin_data_raw.margin_stats[0],
            atol=1 / np.sqrt(SAMPLE_SIZE),
        ).all()
    ):
        print(sample_1p.shape)
        print(np.quantile(sample_1p, q=QTILES, axis=0))
        print(_margin_data_raw.margin_stats[0])
        print(sample_1p.mean())
        raise ValueError("Margin vector sampler does not match hand-rolled sample.")


def test_parallel_data_generation_4() -> None:
    """Test that parallel data generation is replicable."""
    _sseq = SeedSequence(entropy=PPN_S, pool_size=8)

    sample_1p = _margin_resampler(
        PCMDistribution.EMPR_M,
        _margin_data_raw,
        sample_size=(SAMPLE_SIZE, NCOLS),
        seed_sequence=_sseq,
        nthreads=NTHREADS,
    )

    if not np.isclose(
        sample_1p.mean(axis=0),
        _margin_data_raw.margin_stats[0],
        atol=1 / np.sqrt(SAMPLE_SIZE),
    ).all():
        print(sample_1p.shape)
        print(np.quantile(sample_1p, q=QTILES, axis=0))
        print(_margin_data_raw.margin_stats[0])
        print(sample_1p.mean(axis=0))
        raise ValueError("Margin vector sampler does not match hand-rolled sample.")
