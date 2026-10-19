project = "spanormpy"
author = "spanormpy contributors"
extensions = ["myst_parser", "nbsphinx", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
nbsphinx_execute = "never"  # the committed notebook carries its outputs
autosummary_generate = True
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "superpowers", "**.ipynb_checkpoints"]
