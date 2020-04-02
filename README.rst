robupy
======

.. image:: https://zenodo.org/badge/156177030.svg
    :target: https://zenodo.org/badge/latestdoi/156177030

``robupy``  is an open-source Python package for finding worst-case probabilites in the context of robust decision making.
It implements an algorithm, which reduces the selection to a one-dimensional minimization problem. This algorithm was developed
and described in:

    Keane, M. P. and  Wolpin, K. I. (1994). `The Solution and Estimation of Discrete
    Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo
    Evidence <https://doi.org/10.2307/2109768>`_. *The Review of Economics and
    Statistics*, 76(4): 648-672.

You can install ``robupy`` via conda with

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics robupy

Please visit our `online documentation <https://respy.readthedocs.io/en/latest/>`_ for
tutorials and other information.


Citation
--------

If you use robupy for your research, do not forget to cite it with

.. code-block::

    @Unpublished{The robupy team,
            Author = {The robupy team},
            Title  = {robupy - A {P}ython package for robust optimization},
            Year   = {2019},
            Url    = {http://doi.org/10.5281/zenodo.3457403},
    }
