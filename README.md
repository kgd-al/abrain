# Artificial Brains (ABrain) for Python

C++/Python implementation of fully evolvable Artificial Neural Networks.
Uses the ES-HyperNEAT algorithms to *indirectly* encode ANNs with bio-mimetic
patterns (repetitions, symmetry...), large number of neurons and relative
robustness to input/output variations between generations.
The API is served in Python and computations are performed in C++.

[![Documentation Status](https://readthedocs.org/projects/abrain/badge/?version=latest)](https://abrain.readthedocs.io/en/latest/?badge=latest)
![PEP8](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-flake.md)
[![PyPI version](https://badge.fury.io/py/abrain.svg)](https://badge.fury.io/py/abrain)
[![Downloads](https://static.pepy.tech/badge/abrain)](https://pepy.tech/project/abrain)

#### tested on latest
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-wheel-manylinux.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-wheel-musllinux.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-wheel-macosx.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-wheel-win.md)

### Development
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-version.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-tests.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-cov.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-pcov.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-ccov.md)

## Optional dependencies:

### Graphviz (dot)

To generate directed graphs for the genomes.
Can only be fetched by system installer (apt-get, yum, ...).
See https://graphviz.org/download/ for instructions.

.. note::

    On kubuntu 24.04, getting pdf rendering of genomes required installation of `librsvg2-dev/noble graphviz-dev`
    (both?)

### Kaleido

To generate non-interactive images of ANN (through plotly).
Due to inconsistent support, left as an optional dependency.
Use `pip install abrain[...,kaleido]` to get it

## Todo list:
- [ ] Functionalities:
   - [ ] Order-independent ANN evaluation (with back buffer)?
   - [ ] Crossover / historical markings
     - [ ] Actually needed?
   - [ ] MANN Integration
     - [ ] Easy extraction
     - [ ] built-in testing
     - [ ] C++ wrapper
     - [ ] Visu
     
  - [ ] Misc:
    - [ ] Documentation
      - [ ] Advanced usage
      
      - [ ] move to scikit/poetry/... ?

  - [ ] CI/CD
    - [ ] Recent install gives `no loadimage plugion for "svg:cairo"` for pdf output