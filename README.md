# sym2quantized-sapt

[![CI](https://github.com/btyrcha/sym2quantized-sapt/actions/workflows/ci.yml/badge.svg)](https://github.com/btyrcha/sym2quantized-sapt/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21215565.svg)](https://doi.org/10.5281/zenodo.21215565)

A [SymPy](https://www.sympy.org)-based package for second-quantized operator
algebra in **SAPT** (Symmetry-Adapted Perturbation Theory of intermolecular
interactions).

It extends SymPy's `sympy.physics.secondquant` module to work in the **double
Fermi vacuum** — the product of two independent Fermi vacuums, one for each
monomer (A and B) of a dimer. On top of that it provides a generalized Wick's
theorem, dummy-index canonicalization, RHF spin integration, and `numpy.einsum`
code generation, which together let you derive closed-form SAPT energy
expressions symbolically.

## Installation

Python 3.8 is the reference interpreter (it is what CI runs); 3.9-3.12 also
work. `setup.py` is the single source of truth for dependencies - the `dev`
extra adds the test, lint and format tooling.

```shell
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Or with conda:

```shell
conda env create -f environment.yml
conda activate sym2quantized-sapt
```

## Quick start

Derive the first-order SAPT electrostatic energy (`E_pol^{(10)}`):

```python
from sympy import symbols, Dummy, latex
from sym2quantized_sapt.sapt_utils import get_V_operator
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.spin_integrator import spin_integration

V = get_V_operator()
E10 = wicks_double_vac(V, keep_only_fully_contracted=True)
E10 = spin_integration(E10)
print(latex(E10))
```

See the [`examples/`](examples/) directory for full derivations of the
electrostatic, induction, dispersion, and exchange energies.

Starting a project with this package?  [GETTING_STARTED.md](GETTING_STARTED.md)
covers the environment, which branch to start from, the shape of a
derivation script, and the conventions that cost debugging time when
they are violated.

## Running the tests

```shell
# fast tests (example runners and other heavy cases are marked `slow`)
python3 -m pytest ./tests/

# full suite, including the slow tests — note the custom `--slow` flag
python3 -m pytest ./tests/ --slow
```

Many tests assert exact `sympy.latex(...)` output, so they are sensitive to
the SymPy version. `setup.py` caps it at `sympy>=1.8.0,<1.13.0`: 1.13 changed
`Float`/`int` equality and breaks coefficient formatting in `code_generator`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, style checks,
and how derivations under `examples/` are tested.

## License

Distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in academic work, please cite it — see
[CITATION.cff](CITATION.cff). Each GitHub release is archived on Zenodo with a
version-specific DOI; the concept DOI
[10.5281/zenodo.21215565](https://doi.org/10.5281/zenodo.21215565) always
resolves to the latest version. The underlying SAPT formalism follows
S. Rybak, B. Jeziorski, K. Szalewicz,
*J. Chem. Phys.* **95**, 6576 (1991).
