Getting Started
===============

.. note::
    StanBKT cannot be used in the same Python environment as `PyStan` due to 
    conflicts between CmdStanPy and PyStan. If you have PyStan installed, create a 
    separate virtual environment for StanBKT to avoid compatibility issues. 


Requirements
-------------

- Python 3.12 or higher
- `C++` compiler and `make` (For Windows, refer to :ref:`this note <windows_RTools_note>`)
- Supported OS: Windows, macOS, Linux

.. warning::
    **For Windows Users:** StanBKT cannot be installed in a directory that uses `OneDrive`. 
    Specifically users who have their company or institutional 
    accounts listed in the path (e.g. `C:\Users\user\OneDrive - Institution Name (xyz.edu)\Desktop\``).
    This is because CmdStan fails to compile the needed `c++` code in these directories.

Installation
------------

.. tip::
    We **strongly recommend** installing StanBKT in a dedicated virtual environment. 

The recommended way to install StanBKT is using `uv <https://docs.astral.sh/uv/>`__, 
which is a modern Python package manager. You can install StanBKT with the following command:

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: Working with `uv <https://docs.astral.sh/uv/>`__?
        :columns: 12 12 6 6
        :padding: 3

        StanBKT can be installed using the following command:

        ++++++++++++++++++++++

        .. code-block:: bash

            uv add stanbkt

    .. grid-item-card:: Prefer pip?
        :columns: 12 12 6 6
        :padding: 3

        StanBKT can be installed via pip from `PyPI <https://pypi.org/project/stanbkt>`__.

        ++++

        .. code-block:: bash

            pip install stanbkt

Setup
-----
Unlike other common Python packages, StanBKT **requires** additional setup after 
installation, due to its dependencies on `CmdStan <https://github.com/stan-dev/cmdstan>`__. 

StanBKT provides a utility function :func:`stanbkt.utils.setup_cmdstanpy` to automate this setup process, 
which includes installing CmdStan and setting the appropriate environment variables.
The following code should be run after installing StanBKT and only needs to be done 
once per machine.

.. _windows_RTools_note:

.. important:: 
    If you are on Windows, this setup process will install `RTools` which includes the `c++` compiler and `make` binary required for CmdStan. For more details, see the `CmdStanPy installation documentation <https://mc-stan.org/cmdstanpy/installation.html>`__.


.. code-block:: python
    
    from stanbkt.utils import setup_cmdstanpy

    # set up CmdStanPy with 4 cores for parallel processing
    # replace n_cores with the number of cores available
    setup_cmdstanpy(n_cores=4)

