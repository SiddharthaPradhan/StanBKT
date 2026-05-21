<p align="center">
	<img src="docs/source/_static/logo-light.png" alt="StanBKT logo" width="150" />
</p>


# StanBKT
<p align="center">
	<a href="https://stanbkt.readthedocs.io/">
		<img src="https://readthedocs.org/projects/stanbkt/badge/?version=latest" alt="Documentation Status" />
	</a>
	<a href="https://pypi.org/project/stanbkt/">
		<img src="https://img.shields.io/pypi/v/stanbkt" alt="PyPI" />
	</a>
	<a href="LICENSE">
		<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT" />
	</a>
</p>

## Overview
StanBKT is a Python package for fitting Bayesian Knowledge Tracing (BKT) models with Stan (<https://mc-stan.org/>).
It is designed for educational data in long format (student interactions over time) and provides tools for:

- fitting standard and grouped BKT models,
- generating hidden-knowledge predictions,
- simulation utilities for synthetic BKT datasets,
- and plotting/posterior analysis workflows.

## Documentation

Full user docs, API reference, and examples are available at:

<https://stanbkt.readthedocs.io/>

## Installation

StanBKT requires Python 3.12+.

Install from PyPI:

```bash
pip install stanbkt
```

Or with `uv`:

```bash
uv add stanbkt
```

## One-time setup (CmdStan)

After installing StanBKT, run the CmdStan setup once on each machine:

```python
from stanbkt.utils import setup_cmdstanpy

# Adjust n_cores for your machine
setup_cmdstanpy(n_cores=4)
```

This installs/configures CmdStan so model compilation and fitting can run.

## Quick Start

```python
from stanbkt.models import StandardBKT
from stanbkt.utils import sim_simple_BKT

# 1) Generate synthetic interaction data
data = sim_simple_BKT(n_students=100, n_problems=30, n_kcs=3, rng_seed=42)

# 2) Fit a BKT model
model = StandardBKT()
model.fit(data)

# 3) Predict latent knowledge and correctness probabilities
pred = model.predict(data)
print(pred.head())
```

Expected prediction columns include:

- `kc_id`
- `student_id`
- `problem_id`
- `pKnow`
- `pCorrectness`
- `correct`

## Data Format

StanBKT expects interaction data in long format with these columns:

- `student_id`
- `problem_id`
- `correct` (0/1)
- `timestamp` or other ordering field
- `kc_id` (optional; if omitted, all rows are treated as one KC)

If your dataset uses different column names, pass a `column_mapping` dictionary to `fit` and `predict`.

## Notes

- StanBKT uses CmdStanPy under the hood.
- If you use Windows, review the installation notes in the docs for compiler/toolchain setup.
- Running Stan-based inference can take time on first compile.

## License

See the `LICENSE` file.
