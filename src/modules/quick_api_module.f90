!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/16/2020                            !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

! Interface for quick libarary
module quick_api_module

  implicit none
  private

  public :: quick_api
  public :: setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob

#ifdef MPIV
  public :: setQuickMPI
#endif

  type quick_api_type
    ! indicates if quick should run in library mode. This will help
    ! setting up files for quick run.
    logical :: apiMode = .false.

    ! used to determine if memory should be allocated and SAD guess should
    ! be performed
    logical :: firstStep = .true.

    ! current md step, should come from MM code
    integer :: mdstep = 1

    ! keeps track of how many times quick is called by MM code
    integer :: step = 1

    ! number of atoms
    integer :: natoms = 0

    ! number of atom types
    integer :: natm_type = 0

    ! number of external point charges
    integer :: nxt_ptchg = 0

    ! atom type/atomic number of each atom
    integer, allocatable, dimension(:) :: atm_type_id

    ! atomic numbers of atoms
    integer, allocatable, dimension(:) :: atomic_numbers

    ! xyz coordinates of atoms, size is 3*natoms
    double precision, allocatable, dimension(:,:) :: coords

    ! charge and the coordinates of external point charges
    ! size is 4*nxt_ptchg, where last element of a row holds the charge
    double precision, allocatable, dimension(:,:) :: ptchg_crd

    ! job card for quick job, essentially the first line of regular quick input file
    ! default length is 200 characters
    character(len=200) :: keywd

    ! Is the job card provided by passing a string? default is false
    logical :: hasKeywd = .false.

    ! template file name with job card
    character(len=80) :: fqin

    ! if density matrix of the previous step should be used for the current
    ! md step
    logical :: reuse_dmx = .true.

    ! total energy in hartree
    double precision :: tot_ene = 0.0d0

    ! if gradients and point charge gradients are requested
    logical :: isForce = .false.

    ! gradients
    double precision, allocatable, dimension(:,:) :: gradient

    ! point charge gradients
    double precision, allocatable, dimension(:,:) :: ptchg_grad

  end type quick_api_type

! save a quick_api_type varible that othe quick modules can access
  type (quick_api_type), save :: quick_api

#ifdef MPIV
  interface setQuickMPI
    module procedure set_quick_mpi
  end interface
#endif

  interface setQuickJob
    module procedure set_quick_job
  end interface

  interface getQuickEnergy
    module procedure get_quick_energy
  end interface

  interface getQuickEnergyGradients
    module procedure get_quick_energy_gradients
  end interface

  interface deleteQuickJob
    module procedure delete_quick_job
  end interface

contains


! allocates memory for a new quick_api_type variable
subroutine new_quick_api_type(self, natoms, atomic_numbers, ierr)

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer, intent(in)   :: natoms
  integer, intent(in)   :: atomic_numbers(natoms)
  integer, intent(inout)  :: ierr
  integer :: atm_type_id(natoms)
  integer :: i, natm_type

  ! get atom types and number of types
  call get_atom_types(natoms, atomic_numbers, natm_type, atm_type_id, ierr)

  if ( .not. allocated(self%atm_type_id))    allocate(self%atm_type_id(natm_type), stat=ierr)
  if ( .not. allocated(self%atomic_numbers)) allocate(self%atomic_numbers(natoms), stat=ierr)
  if ( .not. allocated(self%coords))         allocate(self%coords(3,natoms), stat=ierr)
  if ( .not. allocated(self%gradient))          allocate(self%gradient(3,natoms), stat=ierr)

 ! save values in the quick_api struct
  self%natoms         = natoms
  self%natm_type      = natm_type
  self%atomic_numbers = atomic_numbers

  do i=1, natm_type
    self%atm_type_id(i) = atm_type_id(i)
  enddo

  ! set result vectors and matrices to zero
  self%gradient  = 0.0d0

end subroutine new_quick_api_type

! this subroutine checks if the string passed through api is a file name
! or a job card
subroutine check_fqin(fqin, keywd, ierr)

  implicit none

  character(len=80), intent(in)  :: fqin
  character(len=200), intent(in) :: keywd
  integer, intent(inout) :: ierr

  call upcase(keywd, 200)

  if ((index(keywd, 'HF') .ne. 0) .or. (index(keywd, 'DFT') .ne. 0) .and. (index(keywd, 'BASIS=') .ne. 0 )) then
    quick_api%hasKeywd = .true.
    quick_api%Keywd = keywd
  endif

  quick_api%fqin    = trim(fqin) // '.in'

end subroutine check_fqin

! reads the job card from template file with .qin extension and initialize quick
! also allocate memory for quick_api internal arrays
subroutine set_quick_job(fqin, keywd, natoms, atomic_numbers, ierr)

  use quick_files_module
  use quick_molspec_module, only : quick_molspec, alloc
  use quick_exception_module
  use quick_method_module

#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none

  character(len=80), intent(in)  :: fqin
  character(len=200), intent(in) :: keywd
  integer, intent(in) :: natoms
  integer, intent(in) :: atomic_numbers(natoms)
  integer, intent(out) :: ierr
  integer :: flen
  ierr=0
  

  ! allocate memory for quick_api_type
  call new_quick_api_type(quick_api, natoms, atomic_numbers, ierr)

  ! check if fqin string is a input file name or job card
  flen = LEN_TRIM(fqin)

  if(flen .gt. 1) then

    quick_api%apiMode = .true.

    call check_fqin(fqin, keywd, ierr)

  endif

  ! Quick calling flow is extremely horrible. Modules are
  ! disorganized and uses stupid tricks to avoid cyclic module
  ! dependency. This must be fixed in future! Being a sheep
  ! for now..

  ! Initialize quick
  call initialize1(ierr)

  ! set the file name and template mode in quick_files_module
  inFileName = quick_api%fqin
  isTemplate = quick_api%apiMode

#ifdef MPIV
  if(master) then
#endif

    ! set quick files
    call set_quick_files(.true.,ierr)
    CHECK_ERROR(ierr)

    ! open output file
    SAFE_CALL(quick_open(iOutFile,outFileName,'U','F','R',.false.,ierr))

    ! print copyright information
    SAFE_CALL(outputCopyright(iOutFile,ierr))

    ! write job information into output file
    SAFE_CALL(PrtDate(iOutFile,'TASK STARTS ON:',ierr))
    call print_quick_io_file(iOutFile,ierr)

#ifdef MPIV
    ! check the mpisize and turn on mpi mode
    !call check_quick_mpi(iOutFile,ierr)

    ! print mpi info into output
    if(bMPI) call print_quick_mpi(iOutFile,ierr)

  endif
#endif

#ifdef CUDA

  ! startup cuda device
  SAFE_CALL(gpu_startup(ierr))

  SAFE_CALL(gpu_set_device(-1,ierr))

  SAFE_CALL(gpu_init(ierr))

  ! write cuda information
  SAFE_CALL(gpu_write_info(iOutFile,ierr))
    !------------------- END CUDA ---------------------------------------
#endif

#ifdef CUDA_MPIV

  SAFE_CALL(mgpu_query(mpisize, mpirank, mgpu_id, ierr))

  SAFE_CALL(mgpu_setup(ierr))

  if(master) SAFE_CALL(mgpu_write_info(iOutFile, mpisize, mgpu_ids, ierr))
  
  SAFE_CALL(mgpu_init(mpirank, mpisize, mgpu_id, ierr))

#endif

  ! read job specifications
  SAFE_CALL(read_Job_and_Atom(ierr))

#if defined CUDA || defined CUDA_MPIV
  call upload(quick_method, ierr)
#endif

  ! save atom number, number of atom types and number of point charges
  ! into quick_molspec
  quick_molspec%natom     = quick_api%natoms
  quick_molspec%iAtomType = quick_api%natm_type

  ! allocate memory for coordinates and charges in molspec
  SAFE_CALL(alloc(quick_molspec,ierr))

end subroutine set_quick_job


! computes atom types
subroutine get_atom_types(natoms, atomic_numbers, natm_type, atm_type_id, ierr)

  implicit none

  integer, intent(in)  :: natoms
  integer, intent(in)  :: atomic_numbers(natoms)
  integer, intent(out) :: natm_type
  integer, intent(out) :: atm_type_id(natoms)
  integer, intent(inout) :: ierr
  integer :: i, j, iatm
  logical :: new_atm_type

  ! set atm_type_id to zero
  call zeroiVec(atm_type_id, natoms)

  ! go through the atomic numbers, find out atom types and save them in
  ! atm_type_id vector
  natm_type = 1
  do i=1, natoms

    new_atm_type = .true.
    iatm = atomic_numbers(i)

    do j=1, natm_type
      if(atm_type_id(j) .eq. iatm) new_atm_type = .false.
    enddo

    if(new_atm_type) then
      atm_type_id(natm_type) = iatm
      natm_type = natm_type+1
    endif

  enddo

  natm_type = natm_type-1

end subroutine get_atom_types

! allocate memory for point charges and gradients
subroutine allocate_point_charge(isgrad,ierr)

  use quick_molspec_module
  use quick_calculated_module
  implicit none
  logical, intent(in) :: isgrad
  integer, intent(inout) :: ierr
  
  ! allocate memory only if external charges exist
  if ( .not. allocated(quick_api%ptchg_crd)) allocate(quick_api%ptchg_crd(4,quick_api%nxt_ptchg), stat=ierr)

  call realloc(quick_molspec,ierr)

  if(isgrad) then
    if ( .not. allocated(quick_api%ptchg_grad)) allocate(quick_api%ptchg_grad(3,quick_api%nxt_ptchg), stat=ierr)
    quick_api%ptchg_grad =0.0d0
    call realloc(quick_qm_struct,ierr)
  endif
  

end subroutine allocate_point_charge

! allocate memory for point charges and gradients
subroutine deallocate_point_charge(isgrad,ierr)

  use quick_molspec_module
  use quick_calculated_module
  implicit none
  logical, intent(in) :: isgrad
  integer, intent(inout) :: ierr

  if ( allocated(quick_api%ptchg_crd))     deallocate(quick_api%ptchg_crd, stat=ierr)

  if(isgrad) then
    if ( allocated(quick_api%ptchg_grad))     deallocate(quick_api%ptchg_grad, stat=ierr)  
  endif

end subroutine deallocate_point_charge

! returns quick qm energy
subroutine get_quick_energy(coords, nxt_ptchg, ptchg_crd, energy, ierr)

  use quick_molspec_module
  implicit none
  
  integer, intent(in)           :: nxt_ptchg
  double precision, intent(in)  :: coords(3,quick_api%natoms)
  double precision, intent(in)  :: ptchg_crd(4,nxt_ptchg)
  double precision, intent(out) :: energy
  integer, intent(out) :: ierr
  ierr=0

  ! assign passed parameter values into quick_api struct
  quick_api%nxt_ptchg = nxt_ptchg
  quick_api%coords        = coords

  ! set number of external atoms in quick_molspec
  quick_molspec%nextatom  = quick_api%nxt_ptchg

  if(quick_api%nxt_ptchg .gt. 0) then
    call allocate_point_charge(.false., ierr)
    quick_api%ptchg_crd     = ptchg_crd
  endif

  call run_quick(quick_api,ierr)

  ! send back total energy and charges
  energy = quick_api%tot_ene

  if(quick_api%nxt_ptchg .gt. 0) call deallocate_point_charge(.false., ierr)

end subroutine get_quick_energy


! calculates and returns energy, gradients and point charge gradients
subroutine get_quick_energy_gradients(coords, nxt_ptchg, ptchg_crd, &
           energy, gradients, ptchg_grad, ierr)

  use quick_molspec_module
  implicit none

  integer, intent(in)             :: nxt_ptchg 
  double precision, intent(in)    :: coords(3,quick_api%natoms)
  double precision, intent(in)    :: ptchg_crd(4,nxt_ptchg)
  double precision, intent(out)   :: energy
  double precision, intent(out)   :: gradients(3,quick_api%natoms)
  double precision, intent(out) :: ptchg_grad(3,nxt_ptchg)
  integer, intent(out) :: ierr
  ierr=0

  ! assign passed parameter values into quick_api struct
  quick_api%coords         = coords
  quick_api%nxt_ptchg = nxt_ptchg

  ! set number of external atoms in quick_molspec
  quick_molspec%nextatom  = quick_api%nxt_ptchg

  if(quick_api%nxt_ptchg .gt. 0) then
    call allocate_point_charge(.true., ierr)
    quick_api%ptchg_crd = ptchg_crd
  endif

  call run_quick(quick_api,ierr)

  ! send back total energy, gradients and point charge gradients
  energy     = quick_api%tot_ene
  gradients     = quick_api%gradient

  if(quick_api%nxt_ptchg .gt. 0) then
    ptchg_grad = quick_api%ptchg_grad
    call deallocate_point_charge(.true., ierr)
  endif

end subroutine get_quick_energy_gradients


! runs quick, partially resembles quick main program
subroutine run_quick(self,ierr)

  use quick_timer_module
  use quick_method_module, only : quick_method
  use quick_files_module
  use quick_calculated_module, only : quick_qm_struct
  use quick_gridpoints_module, only : quick_dft_grid, deform_dft_grid
  use quick_cutoff_module, only: schwarzoff
  use quick_exception_module
  use quick_cshell_eri_module, only: getEriPrecomputables
  use quick_cshell_gradient_module, only: cshell_gradient
  use quick_oshell_gradient_module, only: oshell_gradient
  use quick_optimizer_module, only: optimize
  use quick_sad_guess_module, only: getSadGuess

#ifdef CEW 
  use quick_cew_module, only : quick_cew
#endif

#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer, intent(out) :: ierr
  integer :: i, j, k
  logical :: failed = .false.
  ierr=0

  ! trun off extcharges in quick_method is external charges become zero
  if(quick_api%nxt_ptchg .eq. 0) then
    quick_method%extCharges = .false.
  else
    quick_method%extCharges = .true.
  endif

  ! print step into quick output file
  call print_step(self,ierr)

  ! if dft is requested, make sure to delete dft grid variables from previous
  ! the md step before proceeding
  if(( self%step .gt. 1 ) .and. (quick_method%DFT &
#ifdef CEW
  .or. quick_cew%use_cew &
#endif
   )) then
    call deform_dft_grid(quick_dft_grid)
  endif

  ! set molecular information into quick_molspec
  SAFE_CALL(set_quick_molspecs(quick_api,ierr))

  ! start the timer for initial guess
  call cpu_time(timer_begin%TIniGuess)

  ! we will reuse density matrix for steps above 1. For the 1st step, we should
  ! read basis file and run SAD guess.
  if(self%firstStep .and. self%reuse_dmx) then

    ! perform the initial guess
    if (quick_method%SAD) SAFE_CALL(getSadGuess(ierr))

    ! assign basis functions
    SAFE_CALL(getMol(ierr))

    self%firstStep = .false.

  endif

  ! pre-calculate 2 index coefficients and schwarz cutoff criteria
  if(.not.quick_method%opt) then
    call getEriPrecomputables
    call schwarzoff
  endif

#if defined CUDA || defined CUDA_MPIV
  ! upload molecular and basis information to gpu
  if(.not.quick_method%opt) call gpu_upload_molspecs(ierr)
#endif

  ! stop the timer for initial guess
  call cpu_time(timer_end%TIniGuess)

#ifdef CUDA_MPIV
    timer_begin%T2elb = timer_end%T2elb
    call mgpu_get_2elb_time(timer_end%T2elb)
    timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb
#endif

  timer_cumer%TIniGuess = timer_cumer%TIniGuess+timer_end%TIniGuess-timer_begin%TIniGuess &
                           - (timer_end%T2elb-timer_begin%T2elb)

  ! compute energy
  if ( .not. quick_method%opt .and. .not. quick_method%grad) then
    SAFE_CALL(getEnergy(.false.,ierr))
  endif

  ! compute gradients
  if (.not.quick_method%opt .and. quick_method%grad) then
      if (quick_method%UNRST) then
          SAFE_CALL(oshell_gradient(ierr))
      else
          SAFE_CALL(cshell_gradient(ierr))
      endif
  endif

  ! run optimization
  if (quick_method%opt)  SAFE_CALL(optimize(ierr))

#if defined CUDA || defined CUDA_MPIV
      if (quick_method%bCUDA) then
        call gpu_cleanup()
      endif
#endif

#ifdef MPIV
  if(master) then
#endif

  ! calculate charges
  if (quick_method%dipole) call dipole

! save the results in quick_api struct
  self%tot_ene = quick_qm_struct%Etot

! save gradients and point charge gradients in quick_api struct.
! Note that quick_qm_struct saves both gradients in vector formats and
! we should organize them back into matrix format.
  if (quick_method%grad) then
    k=1
    do i=1,self%natoms
      do j=1,3
        self%gradient(j,i) = quick_qm_struct%gradient(k)
        k=k+1
      enddo
    enddo

    if (quick_method%extCharges .and. self%nxt_ptchg .gt. 0) then
      k=1
      do i=1,self%nxt_ptchg
        do j=1,3
          self%ptchg_grad(j,i) = quick_qm_struct%ptchg_gradient(k)
          k=k+1
        enddo
      enddo
    endif
  endif

#ifdef MPIV
  endif

  ! broadcast results from master to slaves
  call broadcast_quick_mpi_results(self,ierr)
#endif

  ! increase internal quick step by one
  quick_api%step = quick_api%step + 1

end subroutine run_quick


! this subroutine will print the step into quick output file
subroutine print_step(self,ierr)

  use quick_files_module
#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none
  type (quick_api_type) :: self
  integer, intent(inout) :: ierr

  ! print step into quick output file
#ifdef MPIV
  if(master) then
#endif

  write(iOutFile, '(A1)') ' '
  write(iOutFile, '(1x,A16,1x,I12)') '@ Running Step :',self%step
  write(iOutFile, '(A1)') ' '

#ifdef MPIV
  endif
#endif

end subroutine print_step


! this rubroutine will set atom number, types and number of external atoms
! based on the information provided through library api
subroutine set_quick_molspecs(self,ierr)

  use quick_files_module
  use quick_constants_module
  use quick_molspec_module, only : quick_molspec, xyz

  implicit none
  type (quick_api_type) :: self
  integer :: i, j
  integer, intent(inout) :: ierr

  ! pass the step id to quick_files_module
  wrtStep = self%step

  ! save the atom types
  do i=1, self%natm_type
    quick_molspec%atom_type_sym(i) = symbol(self%atm_type_id(i))
  enddo

  ! save the coordinates and atomic numbers
  do i=1, self%natoms
    quick_molspec%iattype(i) = self%atomic_numbers(i)
    do j=1, 3
       xyz(j,i) = self%coords(j,i) * A_TO_BOHRS
    enddo
  enddo

  quick_molspec%xyz => xyz

  ! save the external point charges and coordinates
  if(self%nxt_ptchg .gt. 0) then
    do i=1, self%nxt_ptchg
      do j=1,3
        quick_molspec%extxyz(j,i) = self%ptchg_crd(j,i) * A_TO_BOHRS
      enddo
      quick_molspec%extchg(i)     = self%ptchg_crd(4,i)
    enddo
  endif

end subroutine set_quick_molspecs

#if defined CUDA || defined CUDA_MPIV

! uploads molecular information into gpu
subroutine gpu_upload_molspecs(ierr)

  use quick_molspec_module, only : quick_molspec
  use quick_basis_module

  implicit none
  integer, intent(inout) :: ierr

  call gpu_setup(quick_molspec%natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
       quick_molspec%molchg, quick_molspec%iAtomType)
  call gpu_upload_xyz(quick_molspec%xyz)
  call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)

  call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
  ncontract, itype, aexp, dcoeff, &
  quick_basis%first_basis_function, quick_basis%last_basis_function, &
  quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
  quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
  quick_basis%ktype, quick_basis%kprim, quick_basis%kshell,quick_basis%Ksumtype, &
  quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, quick_basis%Qsbasis, quick_basis%Qfbasis, &
  quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)

  call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)

end subroutine gpu_upload_molspecs

#endif


#ifdef MPIV

! sets mpi variables in quick api
subroutine set_quick_mpi(mpi_rank, mpi_size, ierr)

  use quick_mpi_module

  implicit none

  integer, intent(in) :: mpi_rank, mpi_size
  integer, intent(out) :: ierr

  ! save information in quick_mpi module
  mpirank    = mpi_rank
  mpisize    = mpi_size
  libMPIMode = .true.
  ierr = 0
  

end subroutine set_quick_mpi

! broadcasts results from master to slaves

subroutine broadcast_quick_mpi_results(self,ierr)

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer :: mpierror
  integer, intent(inout) :: ierr

  include 'mpif.h'

  call MPI_BCAST(self%tot_ene,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_BCAST(self%gradient,3*self%natoms,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_BCAST(self%ptchg_grad,3*self%nxt_ptchg,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

end subroutine

#endif

! fialize quick and deallocate memory of quick_api internal arrays
subroutine delete_quick_job(ierr)

  use quick_files_module
  use quick_mpi_module
  use quick_exception_module
  use quick_method_module

  implicit none
  integer, intent(out) :: ierr
  ierr=0

#if defined CUDA || defined CUDA_MPIV
    call delete(quick_method,ierr)
#endif

#ifdef CUDA
  SAFE_CALL(gpu_shutdown(ierr))
#endif

#ifdef CUDA_MPIV
  SAFE_CALL(delete_mgpu_setup(ierr))
  SAFE_CALL(mgpu_shutdown(ierr))
#endif

  ! finalize quick
  call finalize(iOutFile,ierr,1)

  ! deallocate memory
  call delete_quick_api_type(quick_api,ierr)

end subroutine delete_quick_job


! deallocates memory for quick_api_type variable
subroutine delete_quick_api_type(self,ierr)

  implicit none
  type(quick_api_type), intent(inout) :: self
  integer, intent(inout) :: ierr

  if ( allocated(self%atm_type_id))    deallocate(self%atm_type_id, stat=ierr)
  if ( allocated(self%atomic_numbers)) deallocate(self%atomic_numbers, stat=ierr)
  if ( allocated(self%coords))         deallocate(self%coords, stat=ierr)
  if ( allocated(self%gradient))          deallocate(self%gradient, stat=ierr)

end subroutine delete_quick_api_type

end module quick_api_module
