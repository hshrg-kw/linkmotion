# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LinkMotion"
copyright = "2025, hshrg-kw"
author = "hshrg-kw"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Prevent duplicated member documentation
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_typehints = "description"

# Generate autosummary pages
autosummary_generate = True

# Suppress warnings for cross-reference ambiguity and duplicates
suppress_warnings = ["ref.python", "autodoc.duplicate_object"]

# Configure primary references
add_module_names = False
python_use_unqualified_type_names = True

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "_static/logo-light.svg",
    "dark_logo": "_static/logo-dark.svg",
    "github_url": "https://github.com/hshrg-kw/linkmotion",
    "nav_links": [
        {
            "title": "Examples",
            "url": "https://github.com/hshrg-kw/linkmotion/tree/main/examples",
        },
        {
            "title": "Notebooks",
            "url": "https://github.com/hshrg-kw/linkmotion/tree/main/notebooks",
        },
    ],
}

html_title = f"{project}"
html_short_title = project

# Custom CSS
html_css_files = ["custom.css"]
