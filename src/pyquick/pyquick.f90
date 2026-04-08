module pyquick

    use iso_fortran_env, only : output_unit
    use quick_molspec_module, only : quick_molspec, natom, xyz, alloc
    use quick_calculated_module, only : quick_qm_struct
    use quick_method_module, only : quick_method
    use quick_files_module, only : iOutFile, outFileName, inFileName, isTemplate, &
                                   set_quick_files, print_quick_io_file
    use quick_api_module, only : quick_api
    use quick_constants_module, only : SYMBOL, SYMBOL_MAX, A_TO_BOHRS
    use quick_basis_module, only : nbasis, NBSuse
    use quick_cutoff_module, only : schwarzoff
    use quick_eri_cshell_module, only : getEriPrecomputables
    use quick_sad_guess_module, only : getSadGuess

    implicit none

    private
    public :: set_calc, set_basis, set_method, read_geom, &
              had_error, error_message, print_input, input_string, &
              job_run, job_destroy, job_set_output, &
              job_active, &
              job_total_energy, job_e_core, job_e_electronic, &
              job_e_1e, job_e_xc, job_e_disp, &
              job_has_mulliken, job_has_lowdin, job_has_mo_energies, job_has_density_matrix, &
              job_get_mulliken, job_get_lowdin, job_get_mo_energies, job_get_density_matrix

    integer, parameter :: KEYWORD_LEN = 300
    integer, parameter :: INPUT_LEN   = 10000

    character(len=8) :: calc_keyword = ''
    character(len=:), allocatable :: basis_token
    integer, allocatable          :: geom_atnum(:)      ! atomic numbers, length geom_natom
    double precision, allocatable :: geom_coords(:,:)   ! Angstrom coordinates, shape (3, geom_natom)

    logical :: has_calc  = .false.
    logical :: has_basis = .false.
    logical :: has_geom  = .false.

    logical            :: had_error     = .false.
    character(len=512) :: error_message = ''

    character(len=INPUT_LEN) :: input_string = ''

    ! atom count stored by read_geom so job_run knows natom before alloc
    integer :: geom_natom = 0

    ! output file stem (default 'pyquick_job')
    character(len=80) :: output_stem = 'pyquick_job'

    ! job state
    logical :: job_active = .false.

    ! scalar results (always available after a successful job_run)
    double precision :: job_total_energy   = 0.0d0
    double precision :: job_e_core         = 0.0d0
    double precision :: job_e_electronic   = 0.0d0
    double precision :: job_e_1e           = 0.0d0
    double precision :: job_e_xc           = 0.0d0
    double precision :: job_e_disp         = 0.0d0

    ! availability flags for array results
    logical :: job_has_mulliken      = .false.
    logical :: job_has_lowdin        = .false.
    logical :: job_has_mo_energies   = .false.
    logical :: job_has_density_matrix = .false.

    type :: method_entry
        character(len=:), allocatable :: keyword
        character(len=:), allocatable :: arg
    end type method_entry

    type(method_entry), allocatable :: method_list(:)

contains

    ! -----------------------------------------------------------------------
    ! Backward-compatible input-assembly API
    ! -----------------------------------------------------------------------

    subroutine set_calc(keyword)
        character(len=*), intent(in) :: keyword
        character(len=:), allocatable :: normalized

        normalized = uppercase(trim(keyword))

        select case (trim(normalized))
        case ('HF', 'UHF', 'DFT', 'UDFT')
            calc_keyword = normalized
            has_calc = .true.
            call rebuild_input()
        case default
            call fail('set_calc: keyword must be HF, UHF, DFT or UDFT')
        end select
    end subroutine set_calc

    subroutine set_basis(basis_name)
        character(len=*), intent(in) :: basis_name

        if (len_trim(basis_name) == 0) then
            call fail('set_basis: basis name must be non-empty')
            return
        end if

        basis_token = 'BASIS=' // trim(adjustl(basis_name))
        has_basis = .true.
        call rebuild_input()
    end subroutine set_basis

    subroutine set_method(keyword, arg)
        character(len=*), intent(in) :: keyword
        character(len=*), intent(in), optional :: arg
        character(len=:), allocatable :: uname
        integer :: i

        if (len_trim(keyword) == 0) then
            call fail('set_method: keyword name must be non-empty')
            return
        end if

        uname = uppercase(trim(adjustl(keyword)))

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (trim(method_list(i)%keyword) == trim(uname)) then
                    if (present(arg) .and. len_trim(arg) > 0) then
                        method_list(i)%arg = trim(adjustl(arg))
                    else
                        method_list(i)%arg = ''
                    end if
                    call rebuild_input()
                    return
                end if
            end do
        end if

        call append_method(uname, arg)
        call rebuild_input()
    end subroutine set_method

    subroutine read_geom(input)
        character(len=*), intent(in) :: input
        character(len=:), allocatable :: line
        integer :: start, len_input, nl, atom_count, ios, k, z
        character(len=4) :: sym
        double precision :: cx, cy, cz

        ! discard any previous geometry so a second call replaces the first
        if (allocated(geom_atnum))  deallocate(geom_atnum)
        if (allocated(geom_coords)) deallocate(geom_coords)
        geom_natom = 0
        has_geom   = .false.

        ! --- first pass: count non-empty lines to know how many atoms ---
        atom_count = 0
        start      = 1
        len_input  = len(input)

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                if (len_trim(input(start:)) > 0) atom_count = atom_count + 1
                exit
            else
                if (len_trim(input(start:start+nl-2)) > 0) atom_count = atom_count + 1
                start = start + nl
            end if
        end do

        if (atom_count == 0) then
            call fail('read_geom: geometry must contain at least one atom')
            return
        end if

        allocate(geom_atnum(atom_count))
        allocate(geom_coords(3, atom_count))

        ! --- second pass: parse and validate each line ---
        atom_count = 0
        start      = 1

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                line = trim(adjustl(input(start:)))
            else
                line = trim(adjustl(input(start:start+nl-2)))
                start = start + nl
            end if

            if (len_trim(line) == 0) then
                if (nl == 0) exit
                cycle
            end if

            ! parse symbol and three coordinates; accepts decimal and scientific notation
            sym = ''
            read(line, *, iostat=ios) sym, cx, cy, cz
            if (ios /= 0) then
                call fail('read_geom: cannot parse line (expected: SYMBOL X Y Z): ' // &
                          trim(line))
                return
            end if

            ! validate symbol against the QUICK element table
            z = 0
            do k = 1, SYMBOL_MAX
                if (trim(uppercase(sym)) == trim(uppercase(SYMBOL(k)))) then
                    z = k
                    exit
                end if
            end do
            if (z == 0) then
                call fail('read_geom: unknown element symbol "' // trim(sym) // &
                          '" in line: ' // trim(line))
                return
            end if

            atom_count = atom_count + 1
            geom_atnum(atom_count)     = z
            geom_coords(1, atom_count) = cx
            geom_coords(2, atom_count) = cy
            geom_coords(3, atom_count) = cz

            if (nl == 0) exit
        end do

        geom_natom = atom_count
        has_geom   = .true.
        call rebuild_input()

    end subroutine read_geom

    subroutine print_input()
        write(output_unit, '(A)') trim(input_string)
    end subroutine print_input

    subroutine rebuild_input()
        character(len=:), allocatable :: text
        character(len=64) :: atom_line
        integer :: i

        if (has_calc) then
            text = trim(calc_keyword)
        else
            text = ''
        end if

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(text) > 0) then
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                               // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(text) // ' ' // trim(method_list(i)%keyword)
                    end if
                else
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(method_list(i)%keyword) // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(method_list(i)%keyword)
                    end if
                end if
            end do
        end if

        if (has_basis) then
            if (len_trim(text) > 0) then
                text = trim(text) // ' ' // trim(basis_token)
            else
                text = trim(basis_token)
            end if
        end if

        if (has_geom) then
            do i = 1, geom_natom
                write(atom_line, '(A2, 3(1X, F12.6))') &
                    trim(SYMBOL(geom_atnum(i))), &
                    geom_coords(1,i), geom_coords(2,i), geom_coords(3,i)
                text = trim(text) // new_line('a') // trim(atom_line)
            end do
        end if

        input_string = ''
        if (len_trim(text) > 0) then
            input_string(1:min(len_trim(text), INPUT_LEN)) = &
                text(1:min(len_trim(text), INPUT_LEN))
        end if
    end subroutine rebuild_input

    ! -----------------------------------------------------------------------
    ! Job execution API
    ! -----------------------------------------------------------------------

    subroutine job_set_output(stem)
        character(len=*), intent(in) :: stem
        if (len_trim(stem) == 0) then
            call fail('job_set_output: stem must be non-empty')
            return
        end if
        output_stem = trim(stem)
    end subroutine job_set_output

    subroutine job_run()
        ! External free subroutines in libquick.so
        external :: initialize1, read_Job_and_Atom, getMol, getEnergy, dipole
        external :: finalize, outputCopyright, PrtDate, quick_open

        character(len=:), allocatable :: keyword_line
        character(len=10) :: kwlen_str
        integer :: ierr, i, j, k
        integer :: natm_type
        integer :: atm_type_id(geom_natom)
        logical :: new_type
        character(len=256) :: note

        ierr = 0

        ! --- validate prerequisites ---
        if (.not. has_calc) then
            call fail('job_run: call set_calc before run')
            return
        end if
        if (.not. has_basis) then
            call fail('job_run: call set_basis before run')
            return
        end if
        if (.not. has_geom) then
            call fail('job_run: call read_geom before run')
            return
        end if

        ! --- build keyword line ---
        keyword_line = build_keyword_line()

        ! --- guard against silent Fortran truncation on assignment to quick_api%Keywd ---
        if (len_trim(keyword_line) > KEYWORD_LEN) then
            write(kwlen_str, '(I0)') KEYWORD_LEN
            call fail('job_run: keyword line exceeds maximum length of ' // &
                      trim(kwlen_str) // ' characters')
            return
        end if

        ! --- configure quick_api for keyword injection ---
        quick_api%apiMode  = .true.
        quick_api%hasKeywd = .true.
        quick_api%Keywd    = trim(keyword_line)

        ! --- configure file names; isTemplate suppresses coord read in getMol ---
        inFileName = trim(output_stem) // '.in'
        isTemplate = .true.

        ! --- QUICK call chain (follows main.f90) ---
        call initialize1(ierr)
        if (ierr /= 0) then
            call fail('job_run: initialize1 failed')
            return
        end if

        call set_quick_files(.true., ierr)
        if (ierr /= 0) then
            call fail('job_run: set_quick_files failed')
            return
        end if

        call quick_open(iOutFile, outFileName, 'U', 'F', 'R', .false., ierr)
        if (ierr /= 0) then
            call fail('job_run: quick_open failed')
            return
        end if

        call outputCopyright(iOutFile, ierr)
        if (ierr /= 0) then
            call fail('job_run: outputCopyright failed')
            return
        end if

        note = 'TASK STARTS ON:'
        call PrtDate(iOutFile, note, ierr)
        if (ierr /= 0) then
            call fail('job_run: PrtDate failed')
            return
        end if

        call print_quick_io_file(iOutFile, ierr)
        if (ierr /= 0) then
            call fail('job_run: print_quick_io_file failed')
            return
        end if

        ! reads keyword from quick_api%Keywd; skips coordinate read (apiMode)
        call read_Job_and_Atom(ierr)
        if (ierr /= 0) then
            call fail('job_run: read_Job_and_Atom failed')
            return
        end if

        ! set natom (module-level target) BEFORE alloc uses it
        natom = geom_natom

        call alloc(quick_molspec, .false., ierr)
        if (ierr /= 0) then
            call fail('job_run: alloc(quick_molspec) failed')
            return
        end if

        ! --- inject geometry into QUICK module-level state ---

        ! build atom type list (deduplicate by atomic number)
        natm_type = 0
        atm_type_id = 0
        do i = 1, geom_natom
            new_type = .true.
            do k = 1, natm_type
                if (atm_type_id(k) == geom_atnum(i)) then
                    new_type = .false.
                    exit
                end if
            end do
            if (new_type) then
                natm_type = natm_type + 1
                atm_type_id(natm_type) = geom_atnum(i)
            end if
        end do

        quick_molspec%iAtomType = natm_type
        do i = 1, natm_type
            quick_molspec%atom_type_sym(i) = SYMBOL(atm_type_id(i))
        end do

        ! inject atomic numbers and coordinates (convert Angstrom -> Bohr)
        do i = 1, geom_natom
            quick_molspec%iattype(i) = geom_atnum(i)
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do
        quick_molspec%xyz => xyz

        ! --- initial guess ---
        if (quick_method%SAD) then
            call getSadGuess(ierr)
            if (ierr /= 0) then
                call fail('job_run: getSadGuess failed')
                return
            end if
        end if

        ! --- build molecular orbital / basis information ---
        call getMol(ierr)
        if (ierr /= 0) then
            call fail('job_run: getMol failed')
            return
        end if

        ! --- ERI precomputables and cutoff screening ---
        call getEriPrecomputables()
        call schwarzoff()

        ! --- SCF energy ---
        call getEnergy(.false., ierr)
        if (ierr /= 0) then
            call fail('job_run: getEnergy failed')
            return
        end if

        ! --- post-SCF: Mulliken/Lowdin charges and dipole moment ---
        if (quick_method%dipole) call dipole

        ! --- harvest scalar results ---
        job_total_energy   = quick_qm_struct%ETot
        job_e_core         = quick_qm_struct%ECore
        job_e_electronic   = quick_qm_struct%EEl
        job_e_1e           = quick_qm_struct%E1e
        job_e_xc           = quick_qm_struct%Exc
        job_e_disp         = quick_qm_struct%Edisp

        ! --- set array availability flags ---
        job_has_mulliken       = quick_method%dipole
        job_has_lowdin         = quick_method%dipole
        job_has_mo_energies    = allocated(quick_qm_struct%E)
        job_has_density_matrix = allocated(quick_qm_struct%dense)

        job_active = .true.

    end subroutine job_run

    subroutine job_destroy()
        external :: finalize
        integer :: ierr
        ierr = 0
        if (job_active) then
            call finalize(iOutFile, ierr, 1)
            job_active = .false.
        end if
    end subroutine job_destroy

    ! -----------------------------------------------------------------------
    ! Array result getters
    ! Each checks the availability flag and sets had_error if not computed.
    ! -----------------------------------------------------------------------

    subroutine job_get_mulliken(charges, n)
        ! f2py cannot use external module variables as C array bounds, so we
        ! use a literal upper bound and return the actual count in n.
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_mulliken) then
            call fail("'mulliken' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Mulliken(1:natom)
    end subroutine job_get_mulliken

    subroutine job_get_lowdin(charges, n)
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_lowdin) then
            call fail("'lowdin' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Lowdin(1:natom)
    end subroutine job_get_lowdin

    subroutine job_get_mo_energies(energies, n)
        !f2py intent(out) energies, n
        integer, intent(out) :: n
        double precision, intent(out) :: energies(10000)
        energies = 0.0d0
        if (.not. job_has_mo_energies) then
            call fail("'mo_energies' were not computed; run() must complete successfully")
            n = 0
            return
        end if
        n = NBSuse
        energies(1:NBSuse) = quick_qm_struct%E(1:NBSuse)
    end subroutine job_get_mo_energies

    subroutine job_get_density_matrix(dm, nr, nc)
        ! Returns the alpha density matrix as a 1D (row-major) array of length
        ! nr*nc = nbasis*nbasis.  Reshape in Python: dm.reshape(nr, nc).
        ! We cap at 3000*3000 = 9_000_000 elements; large basis sets are rare.
        !f2py intent(out) dm, nr, nc
        integer, intent(out) :: nr, nc
        double precision, intent(out) :: dm(9000000)
        integer :: i, j, idx
        dm = 0.0d0
        if (.not. job_has_density_matrix) then
            call fail("'density_matrix' was not computed; run() must complete successfully")
            nr = 0
            nc = 0
            return
        end if
        nr = nbasis
        nc = nbasis
        do j = 1, nbasis
            do i = 1, nbasis
                idx = (i - 1) * nbasis + j
                dm(idx) = quick_qm_struct%dense(i, j)
            end do
        end do
    end subroutine job_get_density_matrix

    ! -----------------------------------------------------------------------
    ! Private helpers
    ! -----------------------------------------------------------------------

    function build_keyword_line() result(text)
        character(len=:), allocatable :: text
        integer :: i

        text = trim(calc_keyword)

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(method_list(i)%arg) > 0) then
                    text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                           // '=' // trim(method_list(i)%arg)
                else
                    text = trim(text) // ' ' // trim(method_list(i)%keyword)
                end if
            end do
        end if

        if (has_basis) text = trim(text) // ' ' // trim(basis_token)

    end function build_keyword_line

    subroutine append_method(uname, arg)
        character(len=*), intent(in) :: uname
        character(len=*), intent(in), optional :: arg
        type(method_entry), allocatable :: tmp(:)
        integer :: n

        if (.not. allocated(method_list)) then
            allocate(method_list(1))
            n = 1
        else
            n = size(method_list) + 1
            allocate(tmp(n))
            tmp(1:n-1) = method_list
            call move_alloc(tmp, method_list)
        end if

        method_list(n)%keyword = trim(uname)
        if (present(arg) .and. len_trim(arg) > 0) then
            method_list(n)%arg = trim(adjustl(arg))
        else
            method_list(n)%arg = ''
        end if
    end subroutine append_method

    subroutine fail(message)
        character(len=*), intent(in) :: message
        had_error     = .true.
        error_message = trim(message)
    end subroutine fail

    function uppercase(text) result(upper)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: upper
        integer :: i

        upper = text
        do i = 1, len(text)
            select case (upper(i:i))
            case ('a':'z')
                upper(i:i) = achar(iachar(upper(i:i)) - 32)
            end select
        end do
    end function uppercase

end module pyquick
