# pyquick tests (manual, development-time)

These tests are run **manually** during development. They are intentionally **not** wired
into the `runtest` harness or the install process yet — that is deferred to PR time, when
`pyquick_ene_H2O_rhf_sto3g.py` will move to the standard top-level test directory.

## Files
- `pyquick_ene_H2O_rhf_sto3g.py` — reference-energy smoke test (HF/STO-3G water). Exits 0 on
  success, asserts on failure.
- `test_pyquick_api.py` — pytest suite: construction/validation, energy-model identities,
  `Result` gating, and geometry access.

## Prerequisites
1. Build the extension for the Python you'll test with (configure with `-DPYTHON=TRUE`).
2. Make `import pyquick` work — either source the generated `quick.rc` from your install
   (puts `_pyquick*.so` + `__init__.py` in `lib/pyquick` on `PYTHONPATH`), or otherwise
   ensure the built `_pyquick*.so` sits next to `pyquick/__init__.py` on `PYTHONPATH`.
3. Optional: `pip install pytest` to run the suite (and `h5py` if you want to try the
   serialization example in `usecase1.ipynb`).

## Run
```sh
# full suite
pytest src/pyquick/test/test_pyquick_api.py -v

# just the smoke test
python3 src/pyquick/test/pyquick_ene_H2O_rhf_sto3g.py
```
