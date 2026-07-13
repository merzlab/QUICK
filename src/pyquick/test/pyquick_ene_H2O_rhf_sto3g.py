"""
Smoke test for the pyquick Python interface (Calculation/Result API).

Runs an HF/STO-3G energy calculation on water using the same geometry and
cutoff settings as the reference test ene_H2O_rhf_sto3g, then checks that
the total energy matches the saved reference within the standard energy
tolerance used by the QUICK test harness (4.0e-5 Ha).
"""

import math
import sys

try:
    import pyquick
except ImportError as e:
    print(f"FAIL  cannot import pyquick: {e}", file=sys.stderr)
    sys.exit(1)

REF_ENERGY = -74.947863811   # HF/STO-3G total energy for H2O (Ha)
TOL        =   4.0e-5        # standard check_energy threshold from runtest

calc = pyquick.Calculation(
    method='HF',
    basis='STO-3G',
    properties=['mulliken_charges', 'lowdin_charges'],
    keywords={'cutoff': '1.0e-9', 'denserms': '1.0e-6'},
)
result = calc.get_energy('''
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
''')

assert math.isfinite(result.total_energy), \
    f"FAIL  total_energy is not finite: {result.total_energy}"

diff = abs(result.total_energy - REF_ENERGY)
assert diff < TOL, (
    f"FAIL  total_energy {result.total_energy:.9f} Ha differs from reference "
    f"{REF_ENERGY:.9f} Ha by {diff:.2e} (tol {TOL:.0e})"
)

# charges were requested, so they must be available and well-formed
assert len(result.mulliken) == 3, "FAIL  expected 3 Mulliken charges"
assert abs(float(sum(result.mulliken))) < 1.0e-3, \
    f"FAIL  Mulliken charges should sum to ~0 for neutral H2O, got {sum(result.mulliken)}"

print(f"PASS  total_energy = {result.total_energy:.9f} Ha  "
      f"(ref {REF_ENERGY:.9f}, diff {diff:.2e})")
