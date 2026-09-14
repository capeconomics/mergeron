#!/usr/bin/env python3
"""Updates package version number, and commits and tags repository."""

import argparse
import re
import tomllib
from pathlib import Path
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import CalledProcessError
from subprocess import run

import pendulum
import semver

PROJ_DIR = Path(__file__).parent
PKG_SRC_ROOT: Path = next(PROJ_DIR.glob("src/*/__init__.py")).parent
PKG_NAME: str = PKG_SRC_ROOT.name
TSN = pendulum.today()

UV_CMD = Path.home() / ".local" / "bin" / "uv"
GIT_CMD = Path("/usr/bin/git")
PREK_CMD = Path(__file__).parent / ".venv/bin/prek"


def _update_version(_update_level: str) -> None:

    _rc = run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [GIT_CMD, "status", "-uno"], check=True, stdout=PIPE, stderr=STDOUT, text=True
    )
    if "nothing to commit" not in _rc.stdout:
        raise RuntimeError(
            "Repository has uncommitted changes. Commit changes before updating package version."
        )

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
    pkg_init_path = PKG_SRC_ROOT / "__init__.py"
    pkg_init_path.write_text(
        re.sub(
            rf'(?m)^__version__ = "{_pkg_ver}"$',
            f'__version__ = "{_upd_ver}"',
            pkg_init_path.read_text(),
        )
    )

    # Update pyproject.toml
    run([UV_CMD, "version", "--frozen", f"{_upd_ver}"], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]

    # Commit, tag and push
    try:
        _out = run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                GIT_CMD,
                "commit",
                f"{PROJ_DIR / 'docs/source/license.rst'}",
                f"{PROJ_DIR / 'uv.lock'}",
                f"{PROJ_DIR / '.pre-commit-config.yaml'}",
                f"{pkg_init_path}",
                f"{PROJ_DIR / 'pyproject.toml'}",
                "-m",
                f'"chore({TSN.to_date_string()}): update version"',
            ],
            shell=False,
            check=True,
            capture_output=True,
            text=True,
        )
        run([GIT_CMD, "push"], check=True, shell=False)  # ruff: ignore[subprocess-without-shell-equals-true]
        run([GIT_CMD, "tag", f"{_upd_ver}"], check=True, shell=False)  # ruff: ignore[S603]
        run([GIT_CMD, "push", "--tags"], check=True, shell=False)  # ruff: ignore[S603]
    except CalledProcessError as _e:
        print(_out.stdout)
        raise _e


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
