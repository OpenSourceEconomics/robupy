robupy
======
.. image:: https://anaconda.org/opensourceeconomics/robupy/badges/version.svg
    :target: https://anaconda.org/OpenSourceEconomics/robupy

.. image:: https://readthedocs.org/projects/robupy/badge/?version=latest
    :target: https://robupy.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://zenodo.org/badge/156177030.svg
    :target: https://zenodo.org/badge/latestdoi/156177030

``robupy``  is an open-source Python package for finding worst-case probabilites in
the context of robust decision making. It implements an algorithm, which reduces the
selection to a one-dimensional minimization problem. This algorithm was developed and
described in:

    Nilim, A., \& El Ghaoui, L. (2005). `Robust control of Markov decision processes
    with uncertain transition matrices. <https://doi.org/10.1287/opre.1050.0216>`_.
    *Operations Research*, 53(5):  780–798.

You can install ``robupy`` via conda with

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics robupy

Please visit our `online documentation <https://robupy.readthedocs.io/en/latest/>`_ for
tutorials and other information.


Citation
--------

If you use robupy for your research, do not forget to cite it with

.. code-block::

    @Unpublished{The robupy team,
            Author = {The robupy team},
            Title  = {robupy - A {P}ython package for robust optimization},
            Year   = {2019},
            Url    = {https://github.com/OpenSourceEconomics/robupy},
    }
