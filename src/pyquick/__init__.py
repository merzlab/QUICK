try:
    from ._pyquick import pyquick as _mod
except ImportError as e:
    if 'libquick' in str(e):
        raise ImportError(
            "QUICK shared library not found. "
            "Please source the QUICK environment setup script (quick.rc) "
            "before importing this module."
        ) from e
    raise


def _checked(fn):
    """Wrap a Fortran subroutine call; raise RuntimeError if had_error is set."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if _mod.had_error:
            msg = bytes(_mod.error_message).decode().strip()
            _mod.had_error = False
            _mod.error_message = b' ' * 512
            raise RuntimeError(msg)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Module-level backward-compatible API
# ---------------------------------------------------------------------------

set_calc    = _checked(_mod.set_calc)
set_basis   = _checked(_mod.set_basis)
set_method  = _checked(_mod.set_method)
read_geom   = _checked(_mod.read_geom)
print_input = _checked(_mod.print_input)


def __getattr__(name):
    if name == 'input_string':
        return bytes(_mod.input_string).decode().strip()
    raise AttributeError(f"module 'pyquick' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# PyQuick class
# ---------------------------------------------------------------------------

class PyQuick:
    """Object-oriented interface to a single QUICK calculation.

    Usage::

        job = PyQuick()
        job.set_calc('HF')
        job.set_basis('STO-3G')
        job.read_geom('''
            O  0.000  0.000  0.000
            H  0.757  0.586  0.000
            H -0.757  0.586  0.000
        ''')
        job.run()
        print(job.total_energy)
    """

    def __init__(self):
        self._ran = False

    # -- setup methods -------------------------------------------------------

    def set_calc(self, keyword):
        """Set the calculation type: 'HF', 'UHF', 'DFT', or 'UDFT'."""
        _checked(_mod.set_calc)(keyword)

    def set_basis(self, basis_name):
        """Set the basis set, e.g. 'STO-3G', '6-31G*'."""
        _checked(_mod.set_basis)(basis_name)

    def set_method(self, keyword, arg=None):
        """Add or update a keyword token in the job card.

        Examples::

            job.set_method('DIPOLE')          # bare keyword
            job.set_method('CUTOFF', '1e-9')  # keyword=value
        """
        if arg is not None:
            _checked(_mod.set_method)(keyword, arg)
        else:
            _checked(_mod.set_method)(keyword)

    def set_output(self, stem):
        """Set the output file stem (default 'pyquick_job').

        QUICK writes diagnostic output to ``<stem>.out``.
        """
        _checked(_mod.job_set_output)(stem)

    def read_geom(self, geom):
        """Set the molecular geometry.

        *geom* is a multi-line string with one atom per line::

            'SYMBOL  X  Y  Z'

        Coordinates are in Angstrom.
        """
        _checked(_mod.read_geom)(geom)

    def print_input(self):
        """Print the assembled QUICK input to stdout."""
        _checked(_mod.print_input)()

    @property
    def input_string(self):
        """The assembled QUICK input as a string."""
        return bytes(_mod.input_string).decode().strip()

    # -- execution -----------------------------------------------------------

    def run(self):
        """Run the SCF energy calculation.

        Must be called after :meth:`set_calc`, :meth:`set_basis`, and
        :meth:`read_geom`.  Results are available as properties afterwards.
        """
        _checked(_mod.job_run)()
        self._ran = True

    def __del__(self):
        # Only finalize QUICK if this object successfully ran a calculation.
        try:
            if self._ran and _mod.job_active:
                _mod.job_destroy()
        except Exception:
            pass

    # -- scalar results ------------------------------------------------------

    def _require_run(self, prop_name):
        if not self._ran:
            raise AttributeError(
                f"'{prop_name}' is not available until run() has been called"
            )

    @property
    def total_energy(self):
        """Total SCF energy in Hartree."""
        self._require_run('total_energy')
        return float(_mod.job_total_energy)

    @property
    def e_core(self):
        """Core (nuclear repulsion + one-electron) energy in Hartree."""
        self._require_run('e_core')
        return float(_mod.job_e_core)

    @property
    def e_electronic(self):
        """Total electronic energy in Hartree."""
        self._require_run('e_electronic')
        return float(_mod.job_e_electronic)

    @property
    def e_1e(self):
        """One-electron energy in Hartree."""
        self._require_run('e_1e')
        return float(_mod.job_e_1e)

    @property
    def e_xc(self):
        """Exchange-correlation energy in Hartree (0.0 for pure HF)."""
        self._require_run('e_xc')
        return float(_mod.job_e_xc)

    @property
    def e_disp(self):
        """Dispersion correction energy in Hartree (0.0 if not requested)."""
        self._require_run('e_disp')
        return float(_mod.job_e_disp)

    # -- array results -------------------------------------------------------

    def _get_array_result(self, getter_fn, prop_name):
        """Call a Fortran getter and convert RuntimeError to AttributeError."""
        self._require_run(prop_name)
        try:
            return _checked(getter_fn)()
        except RuntimeError as exc:
            raise AttributeError(str(exc)) from None

    @property
    def mulliken(self):
        """Mulliken partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        result, n = self._get_array_result(_mod.job_get_mulliken, 'mulliken')
        return result[:n]

    @property
    def lowdin(self):
        """Lowdin partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        result, n = self._get_array_result(_mod.job_get_lowdin, 'lowdin')
        return result[:n]

    @property
    def mo_energies(self):
        """Molecular orbital energies (alpha) as a numpy array of shape (NBSuse,)."""
        result, n = self._get_array_result(_mod.job_get_mo_energies, 'mo_energies')
        return result[:n]

    @property
    def density_matrix(self):
        """Alpha density matrix as a numpy array of shape (nbasis, nbasis)."""
        result, nr, nc = self._get_array_result(
            _mod.job_get_density_matrix, 'density_matrix'
        )
        # result is a flat 1D array (row-major); reshape to (nr, nc)
        return result[:nr * nc].reshape(nr, nc)
