# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from typing import List, Any, Optional

from sphinx.ext.autodoc import AttributeDocumenter
from sphinx.util import logging

import abrain

# -- Project information -----------------------------------------------------

project = 'ABrain'
copyright = '2022, Kevin Godin-Dubois'
author = 'Kevin Godin-Dubois'

# The full version, including alpha/beta/rc tags
release = abrain.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_design',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = []  # ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'plotly': ('https://plotly.com/python-api-reference/', None)
}

# -- Options for HTML output -------------------------------------------------
autodoc_default_options = {
    "imported-members": True,
    "members": True,
    "member-order": "bysource",
}

# -- Options for Matplotlib autoplot extension --------------------------------

plot_html_show_source_link = False
plot_pre_code = """
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-5,5,100)
"""
plot_html_show_formats = False
plot_rcparams = {
    "figure.autolayout": True,
    "font.size": 24,
    "font.family": "monospace",
}

# -- Options for HTML output -------------------------------------------------

# General API configuration
# object_description_options = [
#     ("py:.*", dict(include_rubrics_in_toc=True)),
# ]

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    # 'page_width': 'auto',
    'nosidebar': False,
    'body_max_width': '100%',
    'enable_search_shortcuts': True,
    'show_relbars': True,

    # 'font_family': 'monospace',

    'github_banner': True,
    'github_button': True,
    'github_repo': 'Py-NeuroEvo',
    'github_user': 'kgd-al',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['custom.css']

# -- Infuriating checker for undocumented items ------------------------------

logger = logging.getLogger(__name__)
kgd_logger = None
kgd_verbose = False

# set up the types of member to check that are documented
members_to_watch = ['function', ]

autodoc_default_options['undoc-members'] = True

keep_warnings = True
nitpicky = True

todo_include_todos = True


def kgd_init_log():
    wd = os.path.dirname(__file__)
    if wd is not None:
        global kgd_logger
        wd += "/_autogen"
        os.makedirs(wd, exist_ok=True)
        kgd_logger = f"{wd}/errors.rst"
        kgd_log(header=True)


def kgd_log(message="", header=False):
    if not header:
        logger.warning(message, type="kgd", subtype="pedantic")
    with open(kgd_logger, mode="w" if header else "a") as log:
        if not header:
            message = f".. warning:: {message}"
        log.write(message + "\n\n")


def contains(pattern: str, docstring: List[str]):
    return any(pattern in ln for ln in docstring)


def simplify(full_namespace, lines, multiline, replacement=""):
    if contains(full_namespace, lines):
        for i in range(len(lines)):
            lines[i] = lines[i].replace(full_namespace, replacement)
        append(lines, '.. note:: Simplified namespace', multiline)


def append(who, what, multiline):
    if multiline:
        who.append(what)
    # else:
    #     who[-1] += what


def process(what, name, obj, lines, multiline):
    if len(lines) == 0:
        kgd_log(f"Undocumented {what} {name} = {obj}({type(obj)})")
        append(lines, f'.. warning:: Undocumented: {what} {name}', multiline)

    # simplify("_cpp.genotype.", lines, multiline)
    simplify("_cpp.phenotype.", lines, multiline)
    simplify("plotly.graph_objs._figure.Figure", lines, multiline,
             "plotly.graph_objs.Figure")

    if kgd_verbose:
        print(f"processing({what}, {name}, {obj}, {lines}, {multiline}")

    if name.endswith("__init__") and len(lines) == 2 and lines[1] == "None":
        del lines[1]

    if contains("arg0", lines):
        kgd_log(f"Warning: Unnamed argument {what} {name} = {obj}({type(obj)})")
        append(lines, f'.. warning:: Unnamed argument: {what} {name}',
               multiline)


def process_docstring(app, what, name, obj, options, lines):
    if kgd_verbose:
        print(f"kgd_autodoc-process-docstring({what}, {name}, {obj}, {lines})")

    process(what, name, obj, lines, True)


def process_signature(app, what, name, obj, options, signature, return_ant):
    ignored = ['class', StaticClassMembersDocumenter.objtype]
    if what in ignored:
        return signature, return_ant

    if kgd_verbose:
        print(f"kgd_autodoc-process-signature({what}, {name}, {obj},"
              f" {signature}, {return_ant})")
    lines = [i for i in [signature, return_ant] if i is not None]
    process(what, name, obj, lines, False)
    r = lines + [None for _ in range(2 - len(lines))]
    return r


class StaticClassMembersDocumenter(AttributeDocumenter):
    objtype = 'kgd_static_member'
    directivetype = "attribute"
    priority = 10 + AttributeDocumenter.priority
    option_spec = dict(AttributeDocumenter.option_spec)

    def get_doc(self) -> Optional[List[List[str]]]:
        r = None
        if hasattr(self.parent, "_docstrings"):
            docs = self.parent._docstrings
            name = self.objpath[-1]
            if name in docs:
                doc = docs[name]
                if "\n" in doc:
                    return [doc.split("\n")]
                else:
                    return [[docs[name]]]
        if r is None:
            r = super().get_doc()
        if kgd_verbose:
            print(f"get_doc(): {r}")
            print(f"\t with {self.parent=}, {self.objpath[-1]=}")
        return r


def setup(app):
    kgd_init_log()
    app.connect('autodoc-process-docstring', process_docstring)
    app.connect('autodoc-process-signature', process_signature)

    app.add_autodocumenter(StaticClassMembersDocumenter)

