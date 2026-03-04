# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))  # package root parent for `import stanbkt`


project = 'StanBKT'
copyright = '2026, Siddhartha Pradhan'
author = 'Siddhartha Pradhan'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # Google/NumPy docstrings
    "sphinx.ext.mathjax",           # :math:`...`
    "sphinx.ext.intersphinx",       # External links for CmdStanPy
    "sphinx_new_tab_link",
    "sphinx_autodoc_typehints", 
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cmdstanpy": ("https://mc-stan.org/cmdstanpy/", None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
