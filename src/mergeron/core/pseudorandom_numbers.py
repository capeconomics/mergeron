"""
Functions for generating synthetic data under specified distributions.

Uses multiple CPUs when available, with PCG64DXSM as the PRNG. [#]_

References
----------

.. [#] See,
    https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html

"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from string import digits
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias
from typing import get_args

import numpy as np
from attrs import Attribute
from attrs import Converter
from attrs import define
from attrs import field
from numpy.random import PCG64DXSM
from numpy.random import Generator
from numpy.random import SeedSequence

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from .. import NTHREADS
from .. import VERSION
from .. import YAML
from .. import ArrayDouble
from .. import ArrayFloat
from .. import yaml_rt_mapper
from . import DEFAULT_BETA_DIST_PARMS
from . import DEFAULT_DIST_PARMS

__version__ = VERSION

RVDistType: TypeAlias = Literal[
    "Beta", "Dirichlet", "Gaussian", "Normal", "Random", "Uniform"
]


def prng(_s: SeedSequence | None = None, /) -> np.random.Generator:
    """Return a psure-random number generator.

    Parameters
    ----------
    _s
        SeedSequence, for generating repeatable, non-overlapping random numbers.

    Returns
    -------
        A numpy random generator.

    """
    return Generator(PCG64DXSM(SeedSequence(pool_size=8) if _s is None else _s))


# Add yaml representer, constructor for SeedSequence
(_, _) = (
    YAML.representer.add_representer(
        SeedSequence, lambda _r, _d: _r.represent_mapping("!SeedSequence", _d.state)
    ),
    YAML.constructor.add_constructor(
        "!SeedSequence", lambda _c, _n, /: SeedSequence(**yaml_rt_mapper(_c, _n))
    ),
)

PPN_S, PPN_L = (
    int(_s)
    for _d, _d0, _d1 in [(digits, digits[0], digits[1:])]
    for _s in (_d1 + _d0 + _d1[::-1], _d1[::-1] + _d)
)


def seed_sequencer(
    _len: int = 3, /, *, generated_entropy: Sequence[int] | None = None
) -> tuple[SeedSequence, ...]:
    R"""
    Return specified number of SeedSequences, for generating random variates.

    Initializes a specified number of SeedSequences based on a set of
    10 generated "seeds" in a hard-coded list. If the required number of
    random variates is larger than 10, the user must first generate
    a sufficient number of seeds to draw upon for initializing SeedSequences.
    The generated entropy can be reused in subsequent calls to this function.

    Parameters
    ----------
    _len
        Number of SeedSequences to initialize

    generated_entropy
        A list of integers with length not less than _s, to be used as seeds
        for initializing SeedSequences. A list of 10 appropriately generated
        integers is used as default.

    Returns
    -------
        A list of numpy SeedSequence objects, which can be used to
        seed prng() or to spawn seed sequences that can be used as seeds for
        generating non-overlapping streams in parallel. [#fn1]_

    Raises
    ------
    ValueError
        When, :math:`\_sseq\_list\_len > max(10, len(generated\_entropy))`.

    References
    ----------
    .. [#fn1] *See*, https://numpy.org/doc/stable/reference/random/parallel.html.
    """
    generated_entropy = generated_entropy or [
        92156365243929466422624541055805800714117298857186959727264899187749727119124,
        45508962760932900824607908382088764294813063250106926349700153055300051503944,
        11358852481965974965852447884047438302274082458147659701772223782670581495409,
        98335771128074178116267837103565107347248838466705856121954317889296202882090,
        99169860978478959086120522268530894898455162069966492625932871292847103049882,
        87208206842095975410011581094164970201731602958127872604742955058753939512957,
        3615645999448046437740316672917313005913548649308233620056831197005377987468,
        108909094416963715978441140822183411647298834317413586830609215654532919223699,
        88096344099146385192471976829122012867254940684757663128881853302534662995332,
        63206306147411023146090085885772240748399174641427012462446714431253444120718,
    ]

    if _len > (_lge := len(generated_entropy)):
        e_str_segs = (
            "This function can presently create SeedSequences for generating up to ",
            f"{_lge:,d} independent random variates. If you really need to generate ",
            f"more than {_lge:,d} seeded independent random variates, please pass a ",
            "sufficiently large list of seeds as generated_entropy. See,",
            "{}/{}.".format(
                "https://numpy.org/doc/stable/reference/random",
                "bit_generators/generated/numpy.random.SeedSequence.html",
            ),
        )
        raise ValueError("".join(e_str_segs))

    return tuple(SeedSequence(_s, pool_size=8) for _s in generated_entropy[:_len])


def _parms_converter(_v: ArrayLike | None, _i: MultithreadedRNG) -> ArrayFloat:
    if (
        _v is None
        or not len(_v)  # type: ignore
        or not (
            _v.any()
            if isinstance(_v, np.ndarray)
            else (_v[0].any() if hasattr(_v, "len") else _v)  # type: ignore[index, union-attr]
        )
    ):
        return {
            "Beta": DEFAULT_BETA_DIST_PARMS,
            "Dirichlet": ArrayFloat(np.ones(_i.values.shape[-1])),
            "Normal": DEFAULT_DIST_PARMS,
            "Uniform": DEFAULT_DIST_PARMS,
        }.get(_i.distribution, DEFAULT_DIST_PARMS)
    elif isinstance(_v, Sequence | np.ndarray):
        return ArrayFloat(np.asarray(_v, float))
    else:
        raise ValueError(
            f"Input, {_v!r} has invalid type. Must be None, Sequence of floats, or Numpy ndarray."
        )


@define
class MultithreadedRNG:
    """Fill given array on demand with pseudo-random numbers as specified.

    Random number generation is multithreaded, using twice
    the number of threads as available CPU cores by default.
    If a seed sequence is provided, it is used in a thread-safe way
    to generate repeatable i.i.d. draws. All arguments are validated
    before commencing multithreaded random number generation.
    """

    values: ArrayDouble = field(kw_only=False, converter=ArrayDouble)
    """Output array to which generated data are over-written

    Array-length defines the number of i.i.d. (vector) draws.
    """

    @values.validator
    def _vsv(
        _i: MultithreadedRNG, _: Attribute[ArrayDouble], _v: ArrayDouble, /
    ) -> None:
        if not len(_v):
            raise ValueError("Output array must at least be one dimensional")

    distribution: RVDistType = field(default="Uniform")
    """Distribution for the generated random numbers.

    Default is "Uniform".
     """

    @distribution.validator
    def _dtv(_i: MultithreadedRNG, _: Attribute[RVDistType], _v: RVDistType, /) -> None:
        if _v not in (_rdts := get_args(RVDistType)):
            raise ValueError(f"Specified distribution must be one of, {_rdts}")

    parameters: ArrayFloat = field(
        converter=Converter(_parms_converter, takes_self=True)  # type: ignore
    )
    """Parameters, if any, for tailoring random number generation."""

    @parameters.default
    def _dpd(self) -> ArrayFloat:
        return _parms_converter(None, self)

    @parameters.validator
    def _dpv(_i: MultithreadedRNG, _: Attribute[ArrayFloat], _v: ArrayFloat, /) -> None:
        if (_i.distribution != "Dirichlet" and (_lrdp := len(_v)) != (_trdp := 2)) or (
            _i.distribution == "Dirichlet"
            and (_lrdp := len(_v)) != (_trdp := _i.values.shape[-1])
        ):
            raise ValueError(f"Expected {_trdp} parameters, got, {_lrdp}")

        elif _i.distribution in {"Beta", "Dirichlet"} and (_v <= 0.0).any():
            raise ValueError("Shape and location parameters must be strictly positive")

    seed_sequence: SeedSequence | None = field(default=None)
    """Seed sequence for generating random numbers."""

    nthreads: int = field(default=NTHREADS)
    """Number of threads to spawn for random number generation."""

    def fill(self) -> None:
        """Fill the provided output array with random number draws as specified.

        This function calls the single-threaded version below.
        """
        if not len(self.parameters) or np.array_equal(
            self.parameters, DEFAULT_DIST_PARMS
        ):
            if self.distribution == "Uniform":
                _distribution: RVDistType = "Random"
            elif self.distribution == "Normal":
                _distribution = "Gaussian"
        else:
            _distribution = self.distribution

        _nthreads = self.nthreads

        _step_size = (len(self.values) / _nthreads).__ceil__()
        _last_size = min(len(self.values), _nthreads * _step_size)

        _seed = (
            SeedSequence(pool_size=8)
            if self.seed_sequence is None
            else self.seed_sequence
        )

        _random_generators = prng(_seed).spawn(_nthreads)

        def _fill_st(
            _rng: Generator,
            _distribution: RVDistType,
            _parameters: ArrayFloat,
            values: ArrayDouble,
            _first: int,
            _last: int,
            /,
        ) -> None:
            """Fill output array using appropriate method (single-threaded)."""
            _sz = values[_first:_last].shape
            match _distribution:
                case "Beta":
                    shape_a, shape_b = _parameters
                    values[_first:_last] = _rng.beta(shape_a, shape_b, size=_sz)
                case "Dirichlet":
                    values[_first:_last] = _rng.dirichlet(_parameters, size=_sz[:-1])
                case "Gaussian":
                    _rng.standard_normal(out=values[_first:_last])
                case "Normal":
                    _mu, _sigma = _parameters
                    values[_first:_last] = _mu + _sigma * _rng.standard_normal(size=_sz)
                case "Random":
                    _rng.random(out=values[_first:_last])
                case "Uniform":
                    _uni_l, _uni_h = _parameters
                    values[_first:_last] = _rng.uniform(_uni_l, _uni_h, size=_sz)
                # case _:
                #     "Unreachable. The validator would have rejected this as invalid."

        with concurrent.futures.ThreadPoolExecutor(_nthreads) as executor_:
            for _i in range(_nthreads):
                executor_.submit(
                    _fill_st,
                    _random_generators[_i],
                    _distribution,  # redefined above from self.distribution
                    self.parameters,
                    self.values,
                    _i * _step_size,
                    _last_size if _i + 1 == _nthreads else (_i + 1) * _step_size,
                )
