robupy
======
.. image:: https://anaconda.org/opensourceeconomics/robupy/badges/version.svg
    :target: https://anaconda.org/OpenSourceEconomics/robupy

.. image:: https://anaconda.org/opensourceeconomics/robupy/badges/platforms.svg
    :target: https://anaconda.org/OpenSourceEconomics/robupy

.. image:: https://readthedocs.org/projects/robupy/badge/?version=latest
    :target: https://robupy.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/OpenSourceEconomics/robupy/workflows/Continuous%20Integration%20Workflow/badge.svg
    :target: https://github.com/OpenSourceEconomics/robupy/actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


``robupy``  is an open-source Python package for finding worst-case probabilities in
the context of robust decision making. It aims to collect algorithms, which find for
different construction methods for the ambiguity set, the worst-case distribution as
fast as possible.

The first algorithm implemented, applies to an ambiguity set constructed with the
Kullback-Leibler divergence function. It reduces the selection to a one-dimensional
minimization problem. This algorithm was developed and described in:

    Nilim, A., \& El Ghaoui, L. (2005). `Robust control of Markov decision processes
    with uncertain transition matrices <https://doi.org/10.1287/opre.1050.0216>`_.
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

.. code-block:: bash

    @Unpublished{robupy.2020,
          author = {{robupy}},
          title  = {A {P}ython package for robust optimization},
          year   = {2020},
          url    = {https://github.com/OpenSourceEconomics/robupy/releases/1.1.1},
        }
