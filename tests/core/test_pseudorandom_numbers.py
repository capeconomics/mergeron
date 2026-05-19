"""Test multithreaded generation of random variates."""

import numpy as np
import numpy.testing as npt

from mergeron.core.pseudorandom_numbers import MultithreadedRNG
from mergeron.core.pseudorandom_numbers import seed_sequencer


def test_mrng_dirichlet(_tcount: int = 10**8, _fcount: int = 5) -> None:
    """Test multithreaded generation of Dirichlet variates."""
    print("Test multithreaded generation of Dirichlet variates")
    test_out = np.empty((_tcount, _fcount), dtype=np.float64)
    mrng_ = MultithreadedRNG(
        test_out,
        distribution="Dirichlet",
        parameters=np.ones(_fcount),
        seed_sequence=seed_sequencer(1)[0],
        nthreads=16,
    )
    mrng_.fill()
    print(test_out)
    print(test_out.mean(axis=0))
    npt.assert_array_equal(
        test_out.mean(axis=0),
        np.array([
            0.1999916675222448,
            0.20000937237277838,
            0.20000280540828835,
            0.2000040263284762,
            0.1999921283682021,
        ]),
    )
    npt.assert_array_almost_equal(
        test_out.mean(axis=0),
        np.array([0.200] * _fcount),
        decimal=int(np.log10(_tcount) / 2),
    )
    npt.assert_array_equal(test_out.shape, (_tcount, _fcount))
    npt.assert_equal(np.round(test_out.sum()), _tcount)
    del test_out, mrng_


def test_mrng_beta(_tcount: int = 10**8, _fcount: int = 5) -> None:
    """Test multithreaded generation of Beta variates."""
    print("Test multithreaded generation of Beta variates")
    test_out = np.empty((_tcount, _fcount), dtype=np.float64)
    mrng_ = MultithreadedRNG(
        test_out,
        distribution="Beta",
        parameters=np.ones(2),
        seed_sequence=seed_sequencer(1)[0],
        nthreads=16,
    )
    mrng_.fill()
    print(test_out.mean(axis=0))
    npt.assert_array_equal(
        test_out.mean(axis=0),
        np.array([
            0.4999797742786088,
            0.5000255089039324,
            0.500004827320672,
            0.5000165032197761,
            0.49997575795924837,
        ]),
    )
    npt.assert_array_almost_equal(
        test_out.mean(axis=0),
        np.array([0.500] * _fcount),
        decimal=int(np.log10(_tcount) / 2),
    )
    npt.assert_array_equal(test_out.shape, (_tcount, _fcount))
    del test_out, mrng_


def test_mrng_beta_scaled(_tcount: int = 10**8, _fcount: int = 5) -> None:
    """Test multithreaded generation of scaled Beta variates."""
    print("Test multithreaded generation of scaled Beta variates")
    test_out = np.empty((_tcount, 1), dtype=np.float64)
    beta_mu, beta_sigma = [0.5, 0.28867513459481287]
    mul = np.divide(beta_mu - beta_mu**2 - beta_sigma**2, beta_sigma**2)
    dist_parms_beta = np.array([beta_mu * mul, (1 - beta_mu) * mul], dtype=np.float64)
    mrng_ = MultithreadedRNG(
        test_out,
        distribution="Beta",
        parameters=dist_parms_beta,
        seed_sequence=seed_sequencer(1)[0],
        nthreads=16,
    )
    mrng_.fill()
    print(test_out.mean())
    npt.assert_allclose(
        test_out.mean(), 0.5000134457423757
    )  # Maintains exact equality intraday but not over several days?
    npt.assert_almost_equal(test_out.mean(), 0.500, decimal=int(np.log10(_tcount) / 2))
    del test_out, mrng_


if __name__ == "__main__":
    test_mrng_dirichlet()
    test_mrng_beta()
    test_mrng_beta_scaled()
