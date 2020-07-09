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
#   include "config.h"

! Program for testing QUICK library API
  program test_quick_api

    use test_quick_api_module, only : loadTestData, printQuickOutput
    use quick_api_module, only : setQuickJob, getQuickEnergy, getQuickEnergyForces, deleteQuickJob 
#ifdef MPIV
    use test_quick_api_module, only : mpi_initialize, printQuickMPIOutput
    use quick_api_module, only : setQuickMPI
#endif

    implicit none

#ifdef MPIV
    include 'mpif.h'
#endif

    ! i, j are some integers useful for loops, frames is the number of
    ! test snapshots (md steps), ierr is for error handling
    integer :: i, j, frames, ierr
   
    ! number of atoms, number of atom types, number of external point charges
    integer :: natoms, nxt_charges

    ! atom type ids, atomic numbers, atomic coordinates, point charges and
    !  coordinates
    integer, allocatable, dimension(:)            :: atomic_numbers 
    double precision, allocatable, dimension(:,:) :: coord          
    double precision, allocatable, dimension(:,:) :: xc_coord       

    ! name of the quick template input file
    character(len=80) :: fname

    ! total qm energy, mulliken charges, forces and point charge gradients
    double precision :: totEne
    double precision, allocatable, dimension(:,:) :: forces         
    double precision, allocatable, dimension(:,:) :: ptchgGrad      

#ifdef MPIV
    ! essential mpi information 
    integer :: mpierror = 0
    integer :: mpirank  = 0
    integer :: mpisize  = 1
    logical :: master   = .true.
#endif

    ierr = 0

#ifdef MPIV
    ! initialize mpi library and get mpirank, mpisize
    call mpi_initialize(mpisize, mpirank, master, mpierror)

    ! setup quick mpi using api, called only once
    call setQuickMPI(mpirank,mpisize)
#endif

    ! set molecule size. We consider a water molecule surounded by 3 point
    ! charges in this test case. 
    natoms      = 3
    nxt_charges = 3    

    ! we consider 5 snapshots of this test system (mimics 5 md steps) 
    frames = 5

    ! alocate memory for input and output arrays. Note that in xc_coord array,
    ! the first 3 columns are the xyz coordinates of the point charges. The
    ! fourth column is the charge. 
    if ( .not. allocated(atomic_numbers)) allocate(atomic_numbers(natoms), stat=ierr) 
    if ( .not. allocated(coord))          allocate(coord(3,natoms), stat=ierr)
    if ( .not. allocated(xc_coord))       allocate(xc_coord(4,nxt_charges), stat=ierr)
    if ( .not. allocated(forces))         allocate(forces(3,natoms), stat=ierr)
    if ( .not. allocated(ptchgGrad))      allocate(ptchgGrad(3,nxt_charges), stat=ierr)

    ! fill up memory with test values, coordinates and external charges will be loded inside 
    ! the loop below.
    fname           = 'water'

    atomic_numbers(1)  = 8
    atomic_numbers(2)  = 1
    atomic_numbers(3)  = 1

    ! set result vectors and matrices to zero
    forces    = 0.0d0
    ptchgGrad = 0.0d0

    ! initialize QUICK, required only once. Assumes keywords for
    ! the QUICK job are provided through a template file.  
    call setQuickJob(fname, natoms, atomic_numbers, nxt_charges)

    do i=1, frames

      ! load coordinates and external point charges for ith step
      call loadTestData(i, natoms, nxt_charges, coord, xc_coord)

      ! compute required quantities, call only a or b. 
      ! a. compute energy
!      call getQuickEnergy(coord, xc_coord, totEne)

      ! b. compute energies, forces and point charge gradients
      call getQuickEnergyForces(coord, xc_coord, totEne, forces, ptchgGrad)    

      ! print values obtained from quick library
#ifdef MPIV
      ! dumb way to sequantially print from all cores..
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      do j=0, mpisize-1
        if(j .eq. mpirank) then
          call printQuickMPIOutput(natoms, nxt_charges, atomic_numbers, totEne, forces, ptchgGrad, mpirank)
        endif
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      enddo 
#else
      call printQuickOutput(natoms, nxt_charges, atomic_numbers, totEne, forces, ptchgGrad)
#endif

    enddo

    ! finalize QUICK, required only once
    call deleteQuickJob()

    ! deallocate memory
    if ( allocated(atomic_numbers)) deallocate(atomic_numbers, stat=ierr)
    if ( allocated(coord))          deallocate(coord, stat=ierr)
    if ( allocated(xc_coord))       deallocate(xc_coord, stat=ierr)
    if ( allocated(forces))         deallocate(forces, stat=ierr)
    if ( allocated(ptchgGrad))      deallocate(ptchgGrad, stat=ierr)

  end program test_quick_api


