"""Configure sphinx documentation."""
# cSpell:disable

import sys
from datetime import datetime
from pathlib import Path
from subprocess import PIPE
from subprocess import run

import semver

version_str = run(
    ["uv", "version"],  # noqa: S607
    stdout=PIPE,
    text=True,
    check=True,
).stdout.strip()

project_name, project_version = version_str.split()
project_version = project_version.replace(".post", "-post")

version_dict = semver.parse(project_version)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = project_name
copyright = f"2017--{datetime.today().year}, S. Murthy Kambhampaty"
author = "S. Murthy Kambhampaty"
version = "{major}.{minor}".format(**version_dict)
release = "{major}.{minor}.{patch}".format(**version_dict)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_immaterial",
    "linuxdoc.rstFlatTable",
    "sphinx.ext.imgconverter",
]

autodoc_typehints = "description"

autoapi_type = "python"
autoapi_add_toctree_entry = True
autoapi_member_order = "source"
autoapi_options = ["members", "undoc-members", "show-inheritance"]
autoapi_template_dir = "_autoapi_templates"
autoapi_keep_files = False

table_styling_embed_css = True

# Latex cutomization
latex_engine = "lualatex"
latex_elements = {
    "fontpkg": R"""
    \setmainfont{Roboto}
    \setsansfont{Roboto}
    \setmonofont{Roboto Mono}
    \setmathfont{STIX Two Math}[math-style=ISO,range={scr,bfscr},StylisticSet=01]
    """,
    "preamble": R"""
    \usepackage[titles]{tocloft}
    \cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
    \setlength{\cftchapnumwidth}{0.75cm}
    \setlength{\cftsecindent}{\cftchapnumwidth}
    \setlength{\cftsecnumwidth}{1.25cm}
    \usepackage[activate={true, nocompatibility}, tracking=true,]{microtype}
    """,
    "fncychap": R"\usepackage[Bjornstrup]{fncychap}",
    "printindex": R"\footnotesize\raggedright\printindex",
}
latex_show_urls = "footnote"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://jbms.github.io/sphinx-immaterial/customization.html
# https://github.com/jbms/sphinx-immaterial/issues/25
html_theme = "sphinx_immaterial"
html_theme_options = {
    "navigation_with_keys": False,
    "features": ["navigation.top", "navigation.tracking", "toc.follow"],
    "toc_title": "Page layout:",
    "site_url": "https://mergeron.capeconomics.com/",
    "repo_url": "https://github.com/capeconomics/mergeron.git",
    "icon": {"repo": "fontawesome/brands/github"},
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/capeconomics/mergeron.git",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/mergeron/",
        },
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "scheme": "default",
            # "toggle": {
            #     "icon": "material/toggle-switch",
            #     "name": "Switch to light mode",
            # },
        }
        # {
        #     "media": "(prefers-color-scheme: light)",
        #     "scheme": "default",
        #     "toggle": {"icon": "material/toggle-switch", "name": "Switch to dark mode"},
        # },
        # {
        #     "media": "(prefers-color-scheme: dark)",
        #     "scheme": "slate",
        #     "toggle": {
        #         "icon": "material/toggle-switch-off-outline",
        #         "name": "Switch to system preference",
        #     },
        # },
    ],
}

html_logo = "_static/output_6_0.png"
html_title = f"{project} {release}"
html_short_title = f"{project}"
html_static_path = ["_static"]

html_css_files = ["css/custom_tables.css"]

# https://stackoverflow.com/questions/70350786/
mathjax3_config = {"chtml": {"displayAlign": "left", "displayIndent": "2em"}}

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, f"{Path.resolve(Path('../../src'))}")
autoapi_dirs = ["../../src/"]
