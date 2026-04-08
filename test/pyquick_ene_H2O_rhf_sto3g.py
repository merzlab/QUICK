"""
Smoke test for the pyquick Python interface.

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

job = pyquick.PyQuick()
job.set_calc('HF')
job.set_basis('STO-3G')
job.set_method('DIPOLE')
job.set_method('cutoff', '1.0e-9')
job.set_method('denserms', '1.0e-6')
job.read_geom('''
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
''')
job.run()

assert math.isfinite(job.total_energy), \
    f"FAIL  total_energy is not finite: {job.total_energy}"

diff = abs(job.total_energy - REF_ENERGY)
assert diff < TOL, (
    f"FAIL  total_energy {job.total_energy:.9f} Ha differs from reference "
    f"{REF_ENERGY:.9f} Ha by {diff:.2e} (tol {TOL:.0e})"
)

print(f"PASS  total_energy = {job.total_energy:.9f} Ha  "
      f"(ref {REF_ENERGY:.9f}, diff {diff:.2e})")
