import io
import sys
from pathlib import Path
from typing import Any

import h5py  # type: ignore
import numpy as np
import pytest
from attrs import fields

from mergeron import YAML
from mergeron import zipfile
from mergeron.core import guidelines_boundaries as gbl
from mergeron.core import pseudorandom_numbers as prn
from mergeron.gen import MarketsData
from mergeron.gen import PCMDistribution
from mergeron.gen import PCMSpec
from mergeron.gen import PriceSpec
from mergeron.gen import SeedSequenceData
from mergeron.gen import ShareSpec
from mergeron.gen import SHRDistribution
from mergeron.gen import UPPTestRegime
from mergeron.gen import data_generation as dgm


@pytest.fixture(scope="session")
def yaml_file_path(request: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pytest fixture to create a path to the test YAML, for writing to and reading from it."""
    return tmp_path_factory.mktemp("test_archive")


@pytest.mark.parametrize("_sample_size", [10**5, 10**7])
def test_serialize_by_hand(_sample_size: int, yaml_file_path: Path) -> None:
    """Test serializing a MarketSample object to file step-by-step.

    The steps involved are hard-coded here.
    """
    seed_data_ = SeedSequenceData(
        *prn.seed_sequencer(2), *[None] * (len(fields(SeedSequenceData)) - 2)
    )

    market_sample = dgm.MarketSample(
        share_spec=ShareSpec(SHRDistribution.UNI, lower_bound=1e-2),
        sample_size=_sample_size,
        seed_data=seed_data_,
        nthreads=16,
    )

    if _sample_size <= 10**6:
        market_sample.generate_sample()
    market_sample.compute_enforcement_counts(
        gbl.GuidelinesStandards(2023).presumption, UPPTestRegime()
    )

    if market_sample.dataset is not None:
        byte_stream = io.BytesIO()
        archive_path = yaml_file_path / "test_ext_serialize_dataset.zip"
        with h5py.File(byte_stream, "w") as h5f:
            for _a in market_sample.dataset.__attrs_attrs__:
                if (
                    _arr := getattr(market_sample.dataset, _a.name)
                ).any() and not np.isnan(_arr).all():
                    h5f[_a.name] = _arr

        with zipfile.ZipFile(archive_path, mode="w", compression=93) as _hzh:
            zpath = zipfile.Path(_hzh)
            with (zpath / "market_sample.yaml").open("w") as _yfh:
                YAML.dump(market_sample, _yfh)

            with (zpath / "market_dataset.h5").open("wb") as _hfh:
                _hfh.write(byte_stream.getvalue())

        with zipfile.ZipFile(archive_path) as _hzf:
            hzp = zipfile.Path(_hzf)
            market_sample_deser = YAML.load((hzp / "market_sample.yaml").read_text())

            with (hzp / "market_dataset.h5").open("rb") as _hfh:
                h5f = h5py.File(_hfh)
                market_dataset = MarketsData(**{_a: h5f[_a][:] for _a in h5f})

        with io.BytesIO() as _byt:
            YAML.dump(market_sample, _byt)
            market_sample_test = YAML.load(_byt.getvalue())

        if market_sample_deser != market_sample_test:
            raise AssertionError(
                "Restored market sample defintion does not match source definition."
            )

        if not all(
            np.array_equal(getattr(market_dataset, _a.name), _arr)
            for _a in market_sample.dataset.__attrs_attrs__
            if (_arr := getattr(market_sample.dataset, _a.name)).any()
            and not np.isnan(_arr).all()
        ):
            raise AssertionError(
                "Restored data sample does not match source data sample."
            )


@pytest.mark.parametrize("_sample_size", [10**5, 10**7])
def test_int_serialize(_sample_size: int, yaml_file_path: Path) -> None:
    """Test serializing a MarketSample object to file, with classmethods."""
    seed_data_ = prn.seed_sequencer(len(fields(SeedSequenceData)))

    market_sample = dgm.MarketSample(
        share_spec=ShareSpec(SHRDistribution.DIR_FLAT, lower_bound=1e-4),
        pcm_spec=PCMSpec(PCMDistribution.BETA),
        price_spec=PriceSpec.PRICE_RND,
        sample_size=_sample_size,
        seed_data=seed_data_,
        nthreads=16,
    )

    if _sample_size <= 10**6:
        market_sample.generate_sample()
    market_sample.compute_enforcement_counts(
        gbl.GuidelinesStandards(2023).presumption, UPPTestRegime()
    )

    archive_path = yaml_file_path / "test_int_serialize.zip"
    archive_subdir = "test_int_serialize"
    with zipfile.ZipFile(archive_path, "w", compression=93) as _yzh:
        market_sample.to_archive(_yzh, archive_subdir, save_dataset=True)

    with zipfile.ZipFile(archive_path, "r") as _yzh:
        market_sample_test = dgm.MarketSample.from_archive(
            _yzh, archive_subdir, restore_dataset=True
        )

    if market_sample_test != market_sample:
        print("Market sample:")
        YAML.dump(market_sample, sys.stdout)
        print(market_sample.enforcement_counts)

        print("Restored market sample:")
        YAML.dump(market_sample_test, sys.stdout)
        print(market_sample_test.enforcement_counts)

        raise AssertionError("Restored market sample does not match source.")
