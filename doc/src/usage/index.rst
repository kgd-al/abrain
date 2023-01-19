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

.. code-block:: console

   $ git clone ...
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

Musllinux wheels
****************

.. note: For tests only

Pillow (used to generate images of CPPN outputs) has no available wheels.
Manual installation requires at least a jpeg library (e.g. `apk add jpeg-dev`)
See https://pillow.readthedocs.io/en/stable/installation.html#building-from-source


