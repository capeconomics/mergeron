"""Test addition of yaml representers and constructors for various objects."""

from __future__ import annotations

import decimal
from pathlib import Path
from typing import TYPE_CHECKING

import mpmath
import numpy as np
import pytest
from attrs import fields
from numpy.random import SeedSequence

import mergeron.core.guidelines_boundaries as gbl
import mergeron.gen.data_generation as dgm
from mergeron import PKG_NAME
from mergeron import WORK_DIR
from mergeron import YAML
from mergeron.gen import DEFAULT_FCOUNT_WTS
from mergeron.gen import PCMDistribution
from mergeron.gen import PCMSpec
from mergeron.gen import PriceSpec
from mergeron.gen import SeedSequenceData
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution

if TYPE_CHECKING:
    from mergeron import MPFloat
    from mergeron import MPMatrix
    from mergeron.core import MGThresholds

if not (_w := Path.home() / PKG_NAME) == WORK_DIR:
    raise ValueError(f"WORK_DIR expected to be {_w} but is {WORK_DIR}")


@pytest.fixture(scope="function")
def yaml_file_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a path to the test folder, for writing to and reading from it."""
    return tmp_path_factory.mktemp("test_yaml") / "test_class_persistence.yaml"


@pytest.mark.parametrize(
    "_test_instance",
    (
        gbl.GuidelinesStandards(1992),
        gbl.GuidelinesStandards(2023).safeharbor,
        ShareSpec(
            SHRDistribution.DIR_FLAT,
            firm_counts_weights=DEFAULT_FCOUNT_WTS,
            parameters=np.ones(1 + len(DEFAULT_FCOUNT_WTS)),
        ),
        SeedSequenceData(*[
            SeedSequence(pool_size=8) for _ in range(len(fields(SeedSequenceData)))
        ]),
        dgm.MarketSample(),
        dgm.MarketSample(
            share_spec=ShareSpec(SHRDistribution.DIR_FLAT),
            pcm_spec=PCMSpec(PCMDistribution.BETA),
            price_spec=PriceSpec.COST_SYM,
        ),
        mpmath.mpf("0.5"),
        mpmath.matrix([["0.5", "1.0"], ["1.0", "2.0"]]),
        [[decimal.Decimal(_j) for _j in _i] for _i in [["1", "2"], ["3", "4"]]],
    ),
)
def test_object_persistence(
    _test_instance: MGThresholds
    | gbl.GuidelinesStandards
    | ShareSpec
    | SeedSequenceData
    | dgm.MarketSample
    | MPFloat
    | MPMatrix
    | list[list[decimal.Decimal]],
    yaml_file_path: Path,
) -> None:
    """Test yaml serialization and deserialization of various objects."""
    test_path = yaml_file_path

    print("The present instance is:")
    print(repr(_test_instance))

    YAML.dump(_test_instance, stream=test_path)
    try:
        test_instance_from_yaml = YAML.load(test_path)
    except NameError:
        print(test_path.read_text())
        print(repr(_test_instance.__class__))
        print(repr(YAML.representer.yaml_representers[_test_instance.__class__]))
        raise

    if _test_instance != test_instance_from_yaml:
        print(repr(_test_instance))
        print(repr(test_instance_from_yaml))
        raise AssertionError(
            "Instances are not equal: {_test_instance_from_yaml!r}\n!=\n{_test_instance!r}"
        )
