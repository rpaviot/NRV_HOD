"""Sphinx configuration for the NRVpy documentation (Read the Docs)."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "NRVpy"
author = "Romain Paviot, Claude (Anthropic)"
copyright = "2026, Romain Paviot"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
# the heavy scientific dependencies are not installed on the docs builder
autodoc_mock_imports = [
    "pyccl", "pycorr", "numba", "interpax", "dark_emulator",
    "nautilus", "iminuit", "fastpt", "flax", "optax", "h5py", "pysco",
    "colossus", "astropy", "matplotlib",
]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "NRVpy"
