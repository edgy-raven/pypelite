import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

import pypelite

project = "pypelite"
author = "pypelite contributors"
release = pypelite.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
toc_object_entries = False
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_extra_path = [".nojekyll"]
html_sidebars = {
    "**": [
        "localtoc.html",
        "relations.html",
        "searchbox.html",
    ]
}
