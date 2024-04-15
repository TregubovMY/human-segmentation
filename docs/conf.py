# Configuration file for the Sphinx documentation builder.
# https://sphinx-ru.readthedocs.io/ru/latest/sphinx.html
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys 
import os

sys.path.insert(0, os.path.abspath(".."))

project = "Human-Segmentation"
copyright = "2024, Tregubov Maksim"
author = "Tregubov Maksim"
release = "1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "press"
# html_theme = "alabaster"

html_static_path = ["_static"]
