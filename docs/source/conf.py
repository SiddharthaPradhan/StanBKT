# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import enum
import stanbkt

project = "StanBKT"
copyright = "2026, Siddhartha Pradhan"
author = "Siddhartha Pradhan"
release = stanbkt.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx.ext.mathjax",  # :math:`...`
    "sphinx.ext.intersphinx",  # External links for CmdStanPy
    "sphinx_new_tab_link",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cmdstanpy": ("https://mc-stan.org/cmdstanpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autosummary_imported_members = True

# TOC tree config
toc_object_entries_show_parents = "hide"

# -- Autodoc configuration ---------------------------------------------------
autodoc_typehints = "description"  # Put type hints in description, not signature
# autodoc_member_order = "bysource"  # Document members in source order
autodoc_default_options = {
    "module": True,
    "show-inheritance": True,
    "undoc-members": True,
    "exclude-members": "maketrans,from_bytes",
}


# Napoleon config
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Suppress specific duplicate object warnings
nitpick_ignore = []

# Suppress cross-reference warnings that arise from symbols being
suppress_warnings = ["ref.python"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "furo"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

# html_theme_options = {
#     "navigation_depth": 1,
# }
title_text = "StanBKT " + str(stanbkt.__version__)
html_theme_options = {
    # Navigation bar
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "text": title_text,
    },
    # Footer config
    "footer_start": ["copyright", "sphinx-version"],
    "footer_center": ["maple-lab"],
    "footer_end": ["theme-version"],
}

# side bar config
html_sidebars = {
    "**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
    "installation": [],
    "index": [],
}
# hide the Show Source link in the right nav bar
html_show_sourcelink = False

# Application Events

# all base classes that contain entries in this list will be ignored.
IGNORE_BASES_LIST = [str, int, enum.StrEnum, enum.IntEnum, object]


def setup(app):
    def shorten_autosummary_titles(app, *args) -> None:
        """Remove module and class from the autosummary titles."""
        autosummary_dir = os.path.join(app.srcdir, "api", "_autosummary")
        if not os.path.exists(autosummary_dir):
            return

        for filename in os.listdir(autosummary_dir):
            if not filename.endswith(".rst"):
                continue

            path = os.path.join(autosummary_dir, filename)
            with open(path, "r") as f:
                lines = f.readlines()

            # skip if missing a title or if a module/class
            if not lines or lines[0].count(".") < 2:
                continue

            short = lines[0].strip().rsplit(".", 1)[-1]
            lines[0] = short + "\n"
            lines[1] = "=" * len(short) + "\n"
            with open(path, "w") as f:
                f.writelines(lines)

    def skip_all_members_from_ignore_list(app, what, name, obj, skip, options):
        # If the member is inherited from any class in IGNORE_BASES_LIST, skip it
        if hasattr(obj, "__objclass__") and any(
            obj.__objclass__ is base for base in IGNORE_BASES_LIST
        ):
            return True
        return skip

    app.connect("autodoc-skip-member", skip_all_members_from_ignore_list)

    # app.connect("autodoc-skip-member", skip_base_members)
    # app.connect("autodoc-process-bases", handle_auto_doc_bases)
    # app.connect("env-before-read-docs", shorten_autosummary_titles)
