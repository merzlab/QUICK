"""
pytest unit + integration tests for the pyquick Calculation/Result API.

Pure-Python tests (validation, method resolution, input assembly) need only that
the compiled extension imports.  The integration tests additionally run a small
HF/STO-3G water job.

Job type is chosen by which method you call:
    calc.get_energy(geometry)  -> single-point energy
    calc.get_grad(geometry)    -> energy + forces        (Phase 2, not yet implemented)
    calc.geo_opt(geometry)     -> optimized geometry      (Phase 2, not yet implemented)
`properties` selects only optional extras (charges, MO energies, density matrix).
"""

import math

import pytest

pyquick = pytest.importorskip("pyquick")


WATER = """
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
"""
REF_ENERGY = -74.947863811
TOL = 4.0e-5


# ---------------------------------------------------------------------------
# Pure-Python: construction & validation
# ---------------------------------------------------------------------------

def test_missing_basis_raises():
    with pytest.raises(ValueError):
        pyquick.Calculation(method='HF', basis='')


def test_unknown_property_raises():
    with pytest.raises(ValueError):
        pyquick.Calculation(method='HF', basis='STO-3G', properties=['bogus'])


def test_properties_default_empty():
    """No properties needed for an energy run; energy is always returned."""
    calc = pyquick.Calculation(method='HF', basis='STO-3G')
    assert calc.properties == set()


def test_jobtype_as_property_is_rejected():
    """energy/gradient/optimize are methods now, so they are not valid property values."""
    for bad in ('energy', 'gradient', 'forces', 'optimize', 'optimized_coords'):
        with pytest.raises(ValueError):
            pyquick.Calculation(method='HF', basis='STO-3G', properties=[bad])


def test_deferred_property_reports_phase():
    with pytest.raises(ValueError) as excinfo:
        pyquick.Calculation(method='HF', basis='STO-3G', properties=['esp_charges'])
    assert 'Phase 4' in str(excinfo.value)


def test_dipole_is_a_supported_property():
    calc = pyquick.Calculation(method='HF', basis='STO-3G', properties=['dipole'])
    assert 'dipole' in calc.properties


@pytest.mark.parametrize("method,mult,expected", [
    ('HF',    1, ('HF', None)),
    ('RHF',   1, ('HF', None)),
    ('HF',    2, ('UHF', None)),
    ('UHF',   1, ('UHF', None)),
    ('B3LYP', 1, ('DFT', 'B3LYP')),
    ('B3LYP', 2, ('UDFT', 'B3LYP')),
    ('pbe0',  1, ('DFT', 'PBE0')),
])
def test_resolve_method(method, mult, expected):
    assert pyquick._resolve_method(method, mult) == expected


def test_resolve_method_dft_without_functional():
    with pytest.raises(ValueError):
        pyquick._resolve_method('DFT', 1)


def test_resolve_method_bad_mult():
    with pytest.raises(ValueError):
        pyquick._resolve_method('HF', 0)


def test_input_string_contains_tokens():
    calc = pyquick.Calculation(
        method='B3LYP', basis='6-31G*',
        properties=['mulliken_charges'],
        charge=-1, mult=2, keywords={'cutoff': '1e-9'})
    s = calc.input_string.upper()
    assert 'UDFT' in s          # functional + open shell -> UDFT
    assert 'B3LYP' in s
    assert 'DIPOLE' in s        # charges -> DIPOLE keyword
    assert 'CHARGE=-1' in s
    assert 'MULT=2' in s
    assert 'CUTOFF=1E-9' in s
    assert 'BASIS=6-31G*' in s


# ---------------------------------------------------------------------------
# geo_opt is still a Phase-2 stub (gradients are implemented; see below)
# ---------------------------------------------------------------------------

def test_geo_opt_not_implemented():
    calc = pyquick.Calculation(method='HF', basis='STO-3G')
    with pytest.raises(NotImplementedError, match="Phase 2"):
        calc.geo_opt(WATER)


# ---------------------------------------------------------------------------
# Integration: get_energy() requires a working SCF run
# ---------------------------------------------------------------------------

def _energy(properties=()):
    calc = pyquick.Calculation(
        method='HF', basis='STO-3G', properties=properties,
        keywords={'cutoff': '1.0e-9', 'denserms': '1.0e-6'})
    return calc.get_energy(WATER)


def test_energy_matches_reference():
    result = _energy()
    assert math.isfinite(result.total_energy)
    assert abs(result.total_energy - REF_ENERGY) < TOL


def test_energy_decomposition_identities():
    """total = nuclear_repulsion + electronic; electronic = 1e + 2e + xc; xc==0 for HF."""
    r = _energy()
    for value in (r.nuclear_repulsion, r.e_electronic, r.e_one_electron,
                  r.e_two_electron, r.e_xc):
        assert math.isfinite(value)
    assert r.total_energy == pytest.approx(r.nuclear_repulsion + r.e_electronic, abs=1e-8)
    assert r.e_electronic == pytest.approx(
        r.e_one_electron + r.e_two_electron + r.e_xc, abs=1e-8)
    assert r.e_xc == pytest.approx(0.0, abs=1e-10)   # pure HF


def test_conditional_energies_gated():
    """dispersion only present if requested; external charge only if EXTCHARGES set.

    The AttributeError must explain *why* the value is unavailable and how to enable it.
    """
    r = _energy()   # no D* keyword, no EXTCHARGES
    with pytest.raises(AttributeError, match="(?i)dispersion"):
        _ = r.e_dispersion
    with pytest.raises(AttributeError, match="EXTCHARGES"):
        _ = r.e_external_charge


def test_unrequested_property_raises_attributeerror():
    """Not-requested properties raise AttributeError naming the property to request."""
    result = _energy()
    with pytest.raises(AttributeError, match="mulliken_charges"):
        _ = result.mulliken
    with pytest.raises(AttributeError, match="density_matrix"):
        _ = result.density_matrix


def test_requested_charges_available():
    result = _energy(['mulliken_charges', 'lowdin_charges'])
    assert len(result.mulliken) == 3
    assert len(result.lowdin) == 3
    assert abs(float(sum(result.mulliken))) < 1.0e-3


def test_density_matrix_shape():
    result = _energy(['density_matrix'])
    dm = result.density_matrix
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1]


def test_geometry_available_from_quick_parse():
    """Geometry (atomic numbers + Angstrom coords) is always exposed, from QUICK's parse."""
    result = _energy()
    assert result.atomic_numbers.tolist() == [8, 1, 1]
    assert result.coordinates.shape == (3, 3)
    # coordinates round-trip the input geometry (Angstrom)
    assert result.coordinates[0] == pytest.approx([-0.33840, 0.00380, 0.23923], abs=1e-4)


# ---------------------------------------------------------------------------
# 2.0 Dipole moment
# ---------------------------------------------------------------------------

def test_dipole_available_when_requested():
    result = _energy(['dipole'])
    d = result.dipole
    assert d.shape == (3,)
    magnitude = math.sqrt(float(sum(x * x for x in d)))   # Debye
    assert 0.3 < magnitude < 4.0            # water is polar (~1.7 D at HF/STO-3G)


def test_dipole_not_requested_raises():
    result = _energy()                       # dipole not in properties
    with pytest.raises(AttributeError, match="dipole"):
        _ = result.dipole


# ---------------------------------------------------------------------------
# 2.1 Nuclear gradient (get_grad) — energy + gradient computed together
# ---------------------------------------------------------------------------

def _grad(properties=()):
    calc = pyquick.Calculation(
        method='HF', basis='STO-3G', properties=properties,
        keywords={'cutoff': '1.0e-11', 'denserms': '1.0e-8'})
    return calc.get_grad(WATER)


def test_get_grad_returns_energy_and_gradient():
    r = _grad()
    # energy is produced by the same job (no separate energy calculation)
    assert abs(r.total_energy - REF_ENERGY) < TOL
    g = r.gradient
    assert g.shape == (3, 3)                  # (natom, 3), Hartree/Bohr
    assert all(math.isfinite(v) for v in g.ravel())
    # translational invariance: the net force on the molecule is ~0
    net = g.sum(axis=0)
    assert max(abs(net)) < 1.0e-4


def test_gradient_unavailable_from_get_energy():
    r = _energy()
    with pytest.raises(AttributeError, match="get_grad"):
        _ = r.gradient


def test_gradient_matches_finite_difference():
    """Central finite differences of the energy reproduce the analytic gradient."""
    ANG_PER_BOHR = 0.52917721067
    h = 0.005  # Angstrom
    atoms = [ln.split() for ln in WATER.strip().splitlines()]

    def energy_at(atom, comp, delta):
        moved = [row[:] for row in atoms]
        moved[atom][1 + comp] = f"{float(moved[atom][1 + comp]) + delta:.6f}"
        geom = "\n".join(" ".join(row) for row in moved)
        calc = pyquick.Calculation(
            method='HF', basis='STO-3G',
            keywords={'cutoff': '1.0e-11', 'denserms': '1.0e-8'})
        return calc.get_energy(geom).total_energy

    g = _grad().gradient          # Hartree/Bohr
    for atom in (0, 1):
        for comp in (0, 1, 2):
            e_plus = energy_at(atom, comp, +h)
            e_minus = energy_at(atom, comp, -h)
            # dE/dx in Hartree/Bohr:  (Ang step) -> (Bohr step)
            fd = (e_plus - e_minus) / (2.0 * h / ANG_PER_BOHR)
            assert g[atom, comp] == pytest.approx(fd, abs=2.0e-3)
