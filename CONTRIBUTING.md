# Contributing to FalcomChain

Thanks for your interest. FalcomChain is an open research library; we welcome
bug reports, fixes, and extensions of the FalCom algorithm.

## Reporting bugs

Open an issue at https://github.com/kirtisoglu/FalcomChain/issues with:

1. **Minimal reproduction** — the smallest piece of code that triggers the bug
2. **Expected vs. actual behavior**
3. **Environment** — Python version, OS, dependency versions
   (`pip freeze | grep -E "(networkx|geopandas|shapely|pandas|numpy)"`)
4. **Random seed** if applicable (chain runs are deterministic with `set_seed()`)

## Development setup

```bash
git clone https://github.com/kirtisoglu/FalcomChain
cd FalcomChain
pip install -e ".[dev]"   # if a dev extra exists, otherwise just -e .
```

## Running tests

```bash
python -m pytest tests/
```

100+ tests should pass. If you add a feature, add a test for it.

## Code style

- Python 3.12+
- Follow the existing style (PEP 8, type hints on public APIs)
- Docstrings on all public classes and functions

## What we welcome

**Bug fixes** are always welcome — open a PR.

**New features** that align with the paper's future-work section:
- Weighted spanning trees
- Adaptive gamma schedules
- New energy functions
- Reversible-ReCom MH correction
- |L| > 2 hierarchy levels
- New ensemble analysis methods

For larger features, please open an issue first to discuss the design.

**Performance improvements** with benchmarks attached.

## What's out of scope

- Visualization changes — those go to the [FalcomPlot](https://github.com/kirtisoglu/FalcomPlot) repo
- Application-specific data loading — keep your domain code in your project,
  not in the library
- Breaking changes to the public API without prior discussion

## Pluggable design

The library is structured so most experiments don't require modifying core
code. Before changing internals, check if you can use:

- A custom `tree_sampler`
- A custom `psi_fn`
- A custom `energy_fn`
- A custom `accept` function
- The `extra_attributes` mechanism for graph data

See [docs/structure.md](docs/structure.md) for the full list of extension points.

## Releasing (maintainers only)

1. Update `CHANGELOG.md` with the new version
2. Bump version in `pyproject.toml`
3. Tag the commit: `git tag -a v0.X.0 -m "Release v0.X.0"`
4. Build: `python -m build`
5. Upload to PyPI: `twine upload dist/*`

## Code of conduct

Be kind and constructive. Open research thrives on respectful disagreement.

## License

By contributing, you agree your contributions are licensed under the MIT License
(see [LICENSE.txt](LICENSE.txt)).
