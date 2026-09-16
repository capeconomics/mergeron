#!/usr/bin/env python3
"""Updates package version number, and commits and tags repository."""

import argparse
import re
import sys
import tomllib
from pathlib import Path
from subprocess import run

import pendulum
import semver

PROJ_DIR = Path(__file__).parent
PKG_INIT_PATH: Path = next(PROJ_DIR.glob("src/*/__init__.py"))
PKG_SRC_ROOT: Path = PKG_INIT_PATH.parent
PKG_NAME: str = PKG_SRC_ROOT.name
TSN = pendulum.today()

UV_CMD = Path.home() / ".local" / "bin" / "uv"
GIT_CMD = "/usr/bin/git"
PREK_CMD = Path(__file__).parent / ".venv/bin/prek"


def _update_version(_update_level: str) -> None:

    # _rc = run(
    #     [GIT_CMD, "status", "-uno"], check=True, capture_output=True, text=True
    # )
    # if "nothing to commit" not in _rc.stdout:
    #     raise RuntimeError(
    #         "Repository has uncommitted changes. Commit changes before updating package version."
    #     )

    _pkg_ver = get_pkg_version()
    _upd_ver = (
        semver.Version(TSN.year, TSN.toordinal(), 0)
        if _update_level == "full"
        else _pkg_ver.bump_patch()
    )

    # Check that package version is older than update version
    if _upd_ver <= _pkg_ver:
        raise ValueError(
            f"Package version, {_pkg_ver} at or above update version, {_upd_ver}. Perhaps update patch-level."
        )

    # Update license
    _license_path = PROJ_DIR / "docs" / "source" / "license.rst"
    _license_path.write_text(
        re.sub(
            r"Copyright (?P<byr>\d{4})-\d{4} (?P<name>S\. Murthy Kambhampaty)",
            rf"Copyright \g<byr>-{TSN.year} \g<name>",
            _license_path.read_text(),
        )
    )

    # Update pagackages/lockfile
    run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            UV_CMD,
            "sync",
            "--active",
            "--all-groups",
            "--no-install-project",
            "--upgrade",
        ],
        check=True,
        shell=False,
    )

    # Update pre-commit hooks
    run([PREK_CMD, "update"], check=True, shell=False)  # ruff: ignore[S603]

    # Update version number in the package file
    PKG_INIT_PATH.write_text(
        re.sub(
            rf'(?m)^__version__ = "{_pkg_ver}"$',
            f'__version__ = "{_upd_ver}"',
            PKG_INIT_PATH.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )

    # Update pyproject.toml
    run([UV_CMD, "version", "--frozen", f"{_upd_ver}"], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]

    # Commit, tag and push
    run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            GIT_CMD,
            "commit",
            "-a",
            # f"{_license_path}",
            # f"{PROJ_DIR / '.pre-commit-config.yaml'}",
            # f"{PROJ_DIR / 'uv.lock'}",
            # f"{PKG_INIT_PATH}",
            # f"{PROJ_DIR / 'pyproject.toml'}",
            "-m",
            f'"chore({TSN.to_date_string()}): update version"',
        ],
        cwd=PROJ_DIR,
        shell=False,
        check=True,
        stdout=sys.stdout,
        text=True,
    )
    run([GIT_CMD, "push"], check=True, shell=False)  # ruff: ignore[subprocess-without-shell-equals-true]
    run([GIT_CMD, "tag", f"{_upd_ver}"], check=True, shell=False)  # ruff: ignore[S603]
    run([GIT_CMD, "push", "--tags"], check=True, shell=False)  # ruff: ignore[S603]


def get_pkg_version() -> semver.Version:
    """Get version number of current package."""
    _t = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    return semver.Version(**semver.parse(_t["project"]["version"]))


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Updates package version number, "
            "commits change, and tags repository. "
            "User must specify `full` or `patch` level update."
        )
    )
    parser.add_argument(
        "update_level",
        type=str,
        choices=["full", "patch"],
        help="Whether `full` or `patch` level version update.",
    )

    args = parser.parse_args()
    _update_version(args.update_level)
