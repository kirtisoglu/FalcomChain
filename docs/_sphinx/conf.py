"""
Sphinx configuration for FalcomChain API documentation.

To build the docs:
    pip install sphinx sphinx-rtd-theme myst-parser
    cd docs/_sphinx
    sphinx-build -b html . _build/html
    open _build/html/index.html
"""

import os
import sys

# Add the project root to sys.path so autodoc can import falcomchain
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information ---------------------------------------------------
project = "FalcomChain"
author = "Hemanshu Kaul, Alaittin Kırtışoğlu"
release = "0.1.0"

# -- General configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # pull docs from docstrings
    "sphinx.ext.napoleon",         # support Google/NumPy docstring styles
    "sphinx.ext.viewcode",         # link to source code
    "sphinx.ext.intersphinx",      # link to networkx, geopandas docs
    "sphinx.ext.autosummary",      # auto-generate summary tables
    "myst_nb",                     # render .md and .ipynb (supersedes myst-parser)
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}
master_doc = "index"

# myst-nb settings
nb_execution_mode = "cache"  # only re-execute when notebook changes
nb_execution_timeout = 120
nb_execution_allow_errors = False

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Cross-references to other libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# -- HTML output ----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_show_sourcelink = False
