Usage
=====

Installation
------------

End-user installation
*********************

.. code-block:: console

   (.venv)$ pip install abrain

Precompiled wheels are available for most linux distributions, macosx and
windows on python 3.8+.
See https://github.com/kgd-al/abrain/wiki/wheels for the full list.


Editable install
****************

Due to inconsistent behavior with pip editable install, it is recommended to
instead clone the repository and use the built dedicated install command:

.. code-block:: console

   $ git clone https://github.com/kgd-al/abrain.git
   $ ./commands.sh install-dev
   OR
   $ ./commands.sh install-dev-tests

Tutorials
---------

.. toctree::
    basics

    advanced/cppn
    advanced/mutations
   
FAQ
---

Windows
*******

Exporting an ANN through plotly requires UTF-8 encoding which is not the
default. Setting an environment variable to `PYTHONUTF8=1` fixes the problem


Kaleido
*******

Rendering an ANN in non-interactive format requires either kaleido or orca.
While the former is unavailable on some distributions, the latter seems out
