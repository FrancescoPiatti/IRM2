"""
Sphinx configuration for the IRM2 documentation.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Put the repo root on sys.path so autodoc can import `src.*`.
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))


# -- Project information -----------------------------------------------------
project = "IRM2"
author = "Imperial College Treasury Modelling Team"
copyright = f"{datetime.now().year}, {author}"
release = "0.1"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",                  # Markdown alongside reST
    "sphinx_autodoc_typehints",     # render type hints from signatures
]

autosummary_generate = True

# Autodoc options.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"   # type-hints under params, not in signature

# Napoleon — render NumPy "Attributes" sections as ``:ivar:`` fields under the
# class signature instead of standalone ``.. attribute::`` directives. Without
# this, dataclass fields end up documented TWICE (once by autodoc from the
# class-scope annotation, once by napoleon from the docstring), producing the
# "duplicate object description" warnings at build time.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

typehints_fully_qualified = False
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch":  ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

exclude_patterns: list[str] = []
templates_path = ["_templates"]

# Suppress the few autodoc / docutils warnings emitted by legacy free-form
# docstring blocks (we keep this short — `napoleon_use_ivar=True` above
# eliminates the bulk of the duplicate-object warnings).
suppress_warnings = [
    "autodoc",
    "docutils",
    "myst",
]


# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "IRM2 — Neural Short-Rate Calibration"
html_short_title = "IRM2"
html_show_sourcelink = True
html_copy_source = False
html_show_sphinx = False

# Sidebar / TOC tuning — deeper nav, no collapse so the full tree is always
# visible when the sidebar is wide enough.
html_theme_options = {
    "navigation_depth": 5,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "includehidden": True,
    "logo_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "style_nav_header_background": "#1f3855",
}

# Pygments code-highlighting style — slightly more contrast than the default.
pygments_style = "default"
