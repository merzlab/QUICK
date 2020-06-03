!---------------------------------------------------------------------!
! Created by Madu Manathunga on 05/29/2020                            !
!                                                                     ! 
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!


! Program for testing QUICK library API
  program test_quick_api

    use test_quick_api_module, only : loadTestData, printQuickOutput
    use quick_api_module, only : setQuickJob, getQuickEnergy, getQuickEnergyForces, deleteQuickJob 
#ifdef QUAPI_MPIV
    use test_quick_api_module, only : mpi_initialize, printQuickMPIOutput
    use quick_api_module, only : setQuickMPI
#endif

    implicit none

#ifdef QUAPI_MPIV
    include 'mpif.h'
#endif

    integer :: i, j,k, frames, ierr
    integer :: natoms, natm_types, nxt_charges
    integer, dimension(:), pointer            :: atom_type_id   => null()
    integer, dimension(:), pointer            :: atomic_numbers => null()
    double precision, dimension(:,:), pointer :: coord          => null()
    double precision, dimension(:,:), pointer :: xc_coord       => null()
    character(len=80) :: fname
    double precision :: totEne
    double precision, dimension(:), pointer :: charge           => null()
    double precision, dimension(:,:), pointer :: forces         => null()
    double precision, dimension(:,:), pointer :: ptchgGrad      => null()
#ifdef QUAPI_MPIV
    integer :: mpierror = 0
    integer :: mpirank  = 0
    integer :: mpisize  = 1
    logical :: master   = .true.
#endif

#ifdef QUAPI_MPIV
    ! initialize mpi library and get mpirank, mpisize
    call mpi_initialize(mpisize, mpirank, master, mpierror)

    ! setup quick mpi using api, called only once
    call setQuickMPI(mpirank,mpisize)
#endif

    ! set molecule size
    natoms      = 3
    natm_types  = 2
    nxt_charges = 3    

    ! alocate memory
    if ( .not. associated(atom_type_id))   allocate(atom_type_id(natm_types), stat=ierr)
    if ( .not. associated(atomic_numbers)) allocate(atomic_numbers(natoms), stat=ierr) 
    if ( .not. associated(coord))          allocate(coord(3,natoms), stat=ierr)
    if ( .not. associated(xc_coord))       allocate(xc_coord(4,nxt_charges), stat=ierr)
    if ( .not. associated(charge))         allocate(charge(natoms), stat=ierr)
    if ( .not. associated(forces))         allocate(forces(3,natoms), stat=ierr)
    if ( .not. associated(ptchgGrad))      allocate(ptchgGrad(3,nxt_charges), stat=ierr)

    ! fill up memory with test values
    fname           = 'water'
    atom_type_id(1) = 8
    atom_type_id(2) = 1

    atomic_numbers(1)  = 8
    atomic_numbers(2)  = 1
    atomic_numbers(3)  = 1

    frames = 5

    ! set result vectors and matrices to zero
    call zeroVec(charge, natoms)
    call zeroMatrix2(forces, natoms, 3)
    call zeroMatrix2(ptchgGrad, nxt_charges, 3)

    ! initialize QUICK, required only once. Assumes keywords for
    ! the QUICK job are provided through a template file.  
    call setQuickJob(fname, natoms, natm_types, nxt_charges)

    do i=1, frames
      call loadTestData(natoms, nxt_charges, coord, xc_coord, i)

      ! compute required quantities, call only a or b. 
      ! a. compute energies and charges
!      call getQuickEnergy(i, atom_type_id, atomic_numbers, coord, xc_coord, totEne, charge)

      ! b. compute energies, charges, gradients and point charge gradients
      call getQuickEnergyForces(i, atom_type_id, atomic_numbers, coord, xc_coord, &
           totEne, charge, forces, ptchgGrad)    

      ! print values obtained from quick library
#ifdef QUAPI_MPIV
      ! dumb way to sequantially print from all cores..
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      do j=0, mpisize-1
        if(j .eq. mpirank) then
          call printQuickMPIOutput(natoms, nxt_charges, atomic_numbers, totEne, charge, forces, ptchgGrad, mpirank)
        endif
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      enddo 
#else
      call printQuickOutput(natoms, nxt_charges, atomic_numbers, totEne, charge, forces, ptchgGrad)
#endif

    enddo

    ! finalize QUICK, required only once
    call deleteQuickJob()

    ! deallocate memory
    if ( associated(atom_type_id))   deallocate(atom_type_id, stat=ierr)
    if ( associated(atomic_numbers)) deallocate(atomic_numbers, stat=ierr)
    if ( associated(coord))          deallocate(coord, stat=ierr)
    if ( associated(xc_coord))       deallocate(xc_coord, stat=ierr)
    if ( associated(charge))         deallocate(charge, stat=ierr)
    if ( associated(forces))         deallocate(forces, stat=ierr)
    if ( associated(ptchgGrad))      deallocate(ptchgGrad, stat=ierr)

  end program test_quick_api


