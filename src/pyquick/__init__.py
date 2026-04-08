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
        self._calc    = None   # str, e.g. 'HF'
        self._basis   = None   # str, e.g. 'STO-3G'
        self._methods = []     # list of (keyword_str, arg_str_or_None)
        self._geom    = None   # raw geometry string as passed by user
        self._ran     = False
        self._results = {}     # snapshot of results captured at end of run()

    # -- setup methods -------------------------------------------------------

    def set_calc(self, keyword):
        """Set the calculation type: 'HF', 'UHF', 'DFT', or 'UDFT'."""
        _checked(_mod.set_calc)(keyword)
        self._calc = keyword

    def set_basis(self, basis_name):
        """Set the basis set, e.g. 'STO-3G', '6-31G*'."""
        _checked(_mod.set_basis)(basis_name)
        self._basis = basis_name

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
        uname = keyword.strip().upper()
        for i, (k, _) in enumerate(self._methods):
            if k == uname:
                self._methods[i] = (uname, arg)
                return
        self._methods.append((uname, arg))

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
        self._geom = geom

    def print_input(self):
        """Print the assembled QUICK input to stdout."""
        print(self.input_string)

    @property
    def input_string(self):
        """The assembled QUICK input as a string."""
        # replay this instance's state so Fortran's input_string reflects it
        _mod.clear_methods()
        if self._calc  is not None: _checked(_mod.set_calc)(self._calc)
        if self._basis is not None: _checked(_mod.set_basis)(self._basis)
        for keyword, arg in self._methods:
            if arg is not None: _checked(_mod.set_method)(keyword, arg)
            else:               _checked(_mod.set_method)(keyword)
        if self._geom  is not None: _checked(_mod.read_geom)(self._geom)
        return bytes(_mod.input_string).decode().strip()

    # -- execution -----------------------------------------------------------

    def run(self):
        """Run the SCF energy calculation.

        Must be called after :meth:`set_calc`, :meth:`set_basis`, and
        :meth:`read_geom`.  Results are available as properties afterwards.
        """
        if self._calc is None:
            raise RuntimeError("call set_calc() before run()")
        if self._basis is None:
            raise RuntimeError("call set_basis() before run()")
        if self._geom is None:
            raise RuntimeError("call read_geom() before run()")
        # replay this instance's state into the Fortran singleton
        _mod.clear_methods()
        _checked(_mod.set_calc)(self._calc)
        _checked(_mod.set_basis)(self._basis)
        for keyword, arg in self._methods:
            if arg is not None:
                _checked(_mod.set_method)(keyword, arg)
            else:
                _checked(_mod.set_method)(keyword)
        _checked(_mod.read_geom)(self._geom)
        _checked(_mod.job_run)()
        self._ran = True
        # snapshot all results into Python-owned storage so that a subsequent
        # run() on a different instance cannot overwrite this instance's results
        self._results['total_energy']   = float(_mod.job_total_energy)
        self._results['e_core']         = float(_mod.job_e_core)
        self._results['e_electronic']   = float(_mod.job_e_electronic)
        self._results['e_1e']           = float(_mod.job_e_1e)
        self._results['e_xc']           = float(_mod.job_e_xc)
        self._results['e_disp']         = float(_mod.job_e_disp)
        if _mod.job_has_mulliken:
            r, n = _mod.job_get_mulliken()
            self._results['mulliken'] = r[:n].copy()
        if _mod.job_has_lowdin:
            r, n = _mod.job_get_lowdin()
            self._results['lowdin'] = r[:n].copy()
        if _mod.job_has_mo_energies:
            r, n = _mod.job_get_mo_energies()
            self._results['mo_energies'] = r[:n].copy()
        if _mod.job_has_density_matrix:
            r, nr, nc = _mod.job_get_density_matrix()
            self._results['density_matrix'] = r[:nr * nc].reshape(nr, nc).copy()

    def copy(self):
        """Return a new PyQuick with the same setup state.

        Results from a previous :meth:`run` are not copied — the new instance
        starts unrun.  All setup attributes (_calc, _basis, _methods, _geom)
        are independent copies, so changes to one object do not affect the other.
        """
        new = PyQuick()
        new._calc    = self._calc
        new._basis   = self._basis
        new._methods = list(self._methods)   # list of immutable tuples — shallow copy is sufficient
        new._geom    = self._geom
        return new

    def __copy__(self):
        return self.copy()

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
        return self._results['total_energy']

    @property
    def e_core(self):
        """Core (nuclear repulsion + one-electron) energy in Hartree."""
        self._require_run('e_core')
        return self._results['e_core']

    @property
    def e_electronic(self):
        """Total electronic energy in Hartree."""
        self._require_run('e_electronic')
        return self._results['e_electronic']

    @property
    def e_1e(self):
        """One-electron energy in Hartree."""
        self._require_run('e_1e')
        return self._results['e_1e']

    @property
    def e_xc(self):
        """Exchange-correlation energy in Hartree (0.0 for pure HF)."""
        self._require_run('e_xc')
        return self._results['e_xc']

    @property
    def e_disp(self):
        """Dispersion correction energy in Hartree (0.0 if not requested)."""
        self._require_run('e_disp')
        return self._results['e_disp']

    # -- array results -------------------------------------------------------

    @property
    def mulliken(self):
        """Mulliken partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        self._require_run('mulliken')
        if 'mulliken' not in self._results:
            raise AttributeError(
                "'mulliken' charges were not computed; "
                "include DIPOLE in the keyword line via set_method('DIPOLE')"
            )
        return self._results['mulliken']

    @property
    def lowdin(self):
        """Lowdin partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        self._require_run('lowdin')
        if 'lowdin' not in self._results:
            raise AttributeError(
                "'lowdin' charges were not computed; "
                "include DIPOLE in the keyword line via set_method('DIPOLE')"
            )
        return self._results['lowdin']

    @property
    def mo_energies(self):
        """Molecular orbital energies (alpha) as a numpy array of shape (NBSuse,)."""
        self._require_run('mo_energies')
        if 'mo_energies' not in self._results:
            raise AttributeError(
                "'mo_energies' were not computed; run() must complete successfully"
            )
        return self._results['mo_energies']

    @property
    def density_matrix(self):
        """Alpha density matrix as a numpy array of shape (nbasis, nbasis)."""
        self._require_run('density_matrix')
        if 'density_matrix' not in self._results:
            raise AttributeError(
                "'density_matrix' was not computed; run() must complete successfully"
            )
        return self._results['density_matrix']
