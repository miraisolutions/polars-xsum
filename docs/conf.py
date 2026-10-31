"""Sphinx configuration of the polars-xsum documentation.

Build with ``uv run --group docs sphinx-build -W --keep-going -n docs docs/_build/html``. The
extension module must be built (``uv sync`` does that), because the API reference is generated
from the docstrings of the installed package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docutils import nodes

from polars_xsum import __version__

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

project = "polars-xsum"
author = "Martin Studer"
copyright = "2026, Mirai Solutions GmbH"
version = release = __version__

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]
exclude_patterns = ["_build"]

# -- API reference -----------------------------------------------------------
autodoc_typehints = "signature"
# Aliases defined under `TYPE_CHECKING` in `polars_xsum`, documented in api.md.
autodoc_type_aliases = {
    "IntoExprColumn": "polars_xsum.IntoExprColumn",
    "ClosedInterval": "polars_xsum.ClosedInterval",
}
python_use_unqualified_type_names = True
numpydoc_show_class_members = False
# Link the types named in "Returns" sections.
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {"pl.Expr": "polars.Expr"}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"polars-xsum {release}"
html_theme_options = {
    "github_url": "https://github.com/miraisolutions/polars-xsum",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {"default_mode": "auto"}
html_show_sourcelink = False
copybutton_exclude = ".linenos, .gp, .go"


# -- Links to Polars classes -------------------------------------------------
# Polars documents `Expr` and `Series` on index pages, without inventory
# entries for the classes themselves. Link the class names to those pages.
_POLARS_API = "https://docs.pola.rs/api/python/stable/reference"
_POLARS_CLASSES = {
    "polars.Expr": f"{_POLARS_API}/expressions/index.html",
    "polars.expr.expr.Expr": f"{_POLARS_API}/expressions/index.html",
    "polars.Series": f"{_POLARS_API}/series/index.html",
    "polars.series.series.Series": f"{_POLARS_API}/series/index.html",
}


def _link_polars_class(
    _app: Sphinx, _env: BuildEnvironment, node: nodes.Element, contnode: nodes.TextElement
) -> nodes.reference | None:
    """Resolve a reference to a Polars class to its page in the Polars docs."""
    uri = _POLARS_CLASSES.get(node.get("reftarget", ""))
    if node.get("refdomain") != "py" or uri is None:
        return None
    return nodes.reference("", "", contnode, internal=False, refuri=uri)


def setup(app: Sphinx) -> None:
    """Register the handler for links to Polars classes."""
    app.connect("missing-reference", _link_polars_class)
