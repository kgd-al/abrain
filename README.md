# Artificial Brains (ABrain) for Python

[![Documentation Status](https://readthedocs.org/projects/abrain/badge/?version=latest)](https://abrain.readthedocs.io/en/latest/?badge=latest)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-flake.md)

### Release
![](https://img.shields.io/badge/on-!-900?logo=pypi)
![](https://img.shields.io/badge/unix-!-900)
![](https://img.shields.io/badge/windows-!-900)

### Development
![](https://img.shields.io/badge/version-!-900)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-tests.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-cov.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-pcov.md)
![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/kgd-al/abrain/badge-ccov.md)


C++/Python implementation of the genotype/phenotype components of Evolvable Substrate HyperNEAT (read without evolution and speciation).

## Optional dependencies:

### Graphviz (dot)

To generate directed graphs for the genomes
Can only be fetched by system installer (apt-get, yum, ...)
See https://graphviz.org/download/ for instructions

### Kaleido

To generate non-interactive images of ANN (through plotly)
Due to inconsistent support, left as an optional dependency
Use `pip install abrain[...,kaleido]` to get it

## Todo list:
 - Functionalities:
   - Order-independent ANN evaluation (with back buffer)?
   - Crossover / historical markings
     - Actually needed?
   - MANN Integration
     - Easy extraction
     - built-in testing
     - C++ wrapper
     - Visu
     
  - Misc:
    - Documentation
      - Usage
      
    - Continuous integration
      - build / tests
      - Badges
      
    - Packaging:
      - include c++ stubs?
      - move to scikit/poetry/... ?
      
    - Installation errors:
      - cp38,...,311-manylinux_i686 fail because cppn outputs diverge between manual and subset/all
        > Caused by unmatched image/host platforms?
      - pp38,39-manylinux_i686,x86_64; pp38,39-macosx_x86_64; pp38,39-win_amd64 fail because of strange import error in numpy
      - cp38,...,311-musllinux_i686 fail because of pillow needs a jpeg library
        > Solved? Added libjpeg to the apk (musllinux) install
        > Removed pillow dependency
      - cp38,..,311-win32,amd64 fail because of encoding error in ANN rendering
