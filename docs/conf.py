# Configuration file for the Sphinx documentation builder.

project = 'cogents-core'
copyright = '2025, Xiaming Chen'
author = 'Xiaming Chen'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme' 