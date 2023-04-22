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

By default these commands will produce a debug build with symbols, suitable
for coverage monitoring. If you want a performance oriented build, instead use:

.. code-block:: console

    $ ./commands.sh install-cached release

Tutorials
---------

.. toctree::
    basics
    evolution

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
