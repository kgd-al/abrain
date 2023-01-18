Usage
=====

Installation
------------

End-user installation
*********************

.. code-block:: console

   (.venv)$ pip install abrain

Precompiled wheels are available (see
https://github.com/kgd-al/abrain/wiki/wheels for the full list)

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

Musllinux wheels
****************

.. note: For tests only

Pillow (used to generate images of CPPN outputs) has no available wheels.
Manual installation requires at least a jpeg library (e.g. `apk add jpeg-dev`)
See https://pillow.readthedocs.io/en/stable/installation.html#building-from-source


