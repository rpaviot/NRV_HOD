"""Sphinx configuration for the NRVpy documentation (Read the Docs)."""

project = "NRVpy"
author = "Romain Paviot, Claude (Anthropic)"
copyright = "2026, Romain Paviot"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "NRVpy"
