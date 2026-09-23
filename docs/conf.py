"""Sphinx configuration for the yg-eo-soilnet documentation.

Build it with ``pixi run -e dev docs``; the pages land in ``docs/_build/html``.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The top-level scripts (main.py, config.py, ...) and the package under src/ are documented from
# the source tree, so they must be importable without installing anything.
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

project = "yg-eo-soilnet"
author = "KhalilM94"
copyright = "2026, KhalilM94"

extensions = [
    "myst_parser",  # pages written in Markdown
    "sphinx.ext.autodoc",  # reference pages built from the docstrings
    "sphinx.ext.napoleon",  # understands NumPy-style docstrings
    "sphinx.ext.viewcode",  # "[source]" links next to every documented object
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# An "Attributes" section becomes a field list on the class rather than a second entry for each
# attribute, which would collide with the property of the same name.
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # Methods scikit-learn adds to every estimator. Their docstrings link into scikit-learn's own
    # documentation, which is not part of this build.
    "exclude-members": (
        "set_fit_request, set_transform_request, set_predict_request, set_score_request, "
        "set_inverse_transform_request, get_metadata_routing"
    ),
}

# The heavy libraries are installed in the pixi environments, so a local build imports the real
# thing. Where they are missing (for example on readthedocs.org) they are replaced by stand-ins, so
# the reference pages still build.
_HEAVY = [
    "torch", "lightning", "mlflow", "sklearn", "xgboost", "shap", "optuna",
    "geopandas", "pyproj", "tabicl", "seaborn", "tqdm", "psutil",
]
autodoc_mock_imports = [name for name in _HEAVY if importlib.util.find_spec(name) is None]

html_theme = "sphinx_rtd_theme"
html_title = "yg-eo-soilnet"
exclude_patterns = ["_build"]
