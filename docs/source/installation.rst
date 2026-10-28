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
- Supported OS: Windows, macOS (Intel and Apple Silicon), Linux (x86_64 and ARM). Intel Macs use numba 0.62.x, the last line with Intel wheels, so they are limited to Python 3.12 and 3.13.

Optional extras
---------------

- ``pip install stanbkt[tbb]`` installs Intel TBB, which numba can use as its threading layer. It is only
  available on x86_64 Linux and x86_64 Windows and is not required; numba falls back to OpenMP or a serial path
  for the parallel posterior summaries when TBB is missing.
- GPU: Stan can offload some computation through OpenCL. Compile with
  ``cpp_compile_kwargs={"STAN_OPENCL": True}`` and select a device with the ``opencl_ids=(platform, device)``
  fit option. This needs an OpenCL runtime and drivers. The BKT likelihood is a custom recursion that OpenCL
  does not accelerate, so for faster fits use CPU threads instead: compile with
  ``cpp_compile_kwargs={"STAN_THREADS": True}`` and set ``threads_per_chain``.
- Numba caches compiled kernels next to the package. On a read only install set ``NUMBA_CACHE_DIR`` to a writable folder.

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


.. note::
    ``setup_cmdstanpy`` downloads CmdStan from GitHub and raises a ``RuntimeError`` if the download fails.
    On shared networks (CI runners, university networks) GitHub may rate limit the request. Set the
    ``GITHUB_PAT`` environment variable to a GitHub token and run it again.

Pinning a CmdStan version
^^^^^^^^^^^^^^^^^^^^^^^^^

By default ``setup_cmdstanpy`` installs the latest CmdStan release. For reproducible analyses
(for example, results reported in a paper), pin the version instead. It is installed next to
any existing versions if it is missing:

.. code-block:: python

    from stanbkt.utils import setup_cmdstanpy

    setup_cmdstanpy(n_cores=4, version="2.37.0")

In a new Python session CmdStanPy uses the newest installed CmdStan, so call
``setup_cmdstanpy(version=...)`` at the start of every script or notebook that should use the
pinned version. Once that version is installed the call returns immediately. Check the active
version with ``cmdstanpy.cmdstan_version()``.
