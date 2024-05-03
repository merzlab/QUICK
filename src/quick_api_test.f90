#include "util.fh"
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
    use quick_api_module, only : setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob 
    use quick_exception_module
#ifdef MPIV
    use test_quick_api_module, only : mpi_initialize, printQuickMPIOutput, mpi_exit
    use quick_api_module, only : setQuickMPI
#endif
#ifdef MPIV
    use mpi
#endif

    implicit none

    ! i, j are some integers useful for loops, frames is the number of
    ! test snapshots (md steps), ierr is for error handling
    integer :: i, j, frames, ierr
   
    ! number of atoms, number of atom types, number of external point charges
    integer :: natoms, nxt_charges

    ! whether to reuse density matrix during MD
    logical :: reuse_dmx

    ! atom type ids, atomic numbers, atomic coordinates, point charges and
    !  coordinates
    integer, allocatable, dimension(:)            :: atomic_numbers 
    double precision, allocatable, dimension(:,:) :: coord          
    double precision, allocatable, dimension(:,:) :: xc_coord       

    ! name of the quick template input file
    character(len=80) :: fname

    ! job card
    character(len=256) :: keywd

    ! total qm energy, mulliken charges, gradients and point charge gradients
    double precision :: totEne
    double precision, allocatable, dimension(:,:) :: gradients         
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
    call setQuickMPI(mpirank,mpisize,ierr)
    CHECK_ERROR(ierr)
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
    if ( .not. allocated(gradients))         allocate(gradients(3,natoms), stat=ierr)
    CHECK_ERROR(ierr)

    ! fill up memory with test values, coordinates and external charges will be loded inside 
    ! the loop below.
    fname           = 'api_water_rhf_631g'
    keywd           = 'HF BASIS=6-31G CUTOFF=1.0D-10 DENSERMS=1.0D-6 GRADIENT EXTCHARGES'
    !keywd =''

    atomic_numbers(1)  = 8
    atomic_numbers(2)  = 1
    atomic_numbers(3)  = 1

    ! set result vectors and matrices to zero
    gradients    = 0.0d0

    ! reuse density matrix during MD
    reuse_dmx=.true.

    ! initialize QUICK, required only once. Assumes keywords for
    ! the QUICK job are provided through a template file.  
    call setQuickJob(fname, keywd, natoms, atomic_numbers, reuse_dmx, ierr)
    CHECK_ERROR(ierr)

    do i=1, frames

      ! load coordinates and external point charges for ith step
      nxt_charges = mod(i,4)

      if ( .not. allocated(xc_coord)) allocate(xc_coord(4,nxt_charges), stat=ierr)      
      if ( .not. allocated(ptchgGrad)) allocate(ptchgGrad(3,nxt_charges), stat=ierr)
      CHECK_ERROR(ierr)

      call loadTestData(i, natoms, nxt_charges, coord, xc_coord) 

      ptchgGrad = 0.0d0

      ! compute required quantities, call only a or b. 
      ! a. compute energy
!      call getQuickEnergy(coord, nxt_charges, xc_coord, totEne)

      ! b. compute energies, gradients and point charge gradients
      call getQuickEnergyGradients(coord, nxt_charges, xc_coord, &
         totEne, gradients, ptchgGrad, ierr)    
      CHECK_ERROR(ierr)

      ! print values obtained from quick library
#ifdef MPIV
      ! dumb way to sequantially print from all cores..
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      do j=0, mpisize-1
        if(j .eq. mpirank) then
          call printQuickMPIOutput(natoms, nxt_charges, atomic_numbers, totEne, gradients, ptchgGrad, mpirank)
        endif
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      enddo 
#else
      call printQuickOutput(natoms, nxt_charges, atomic_numbers, totEne, gradients, ptchgGrad)
#endif

      if ( allocated(xc_coord))       deallocate(xc_coord, stat=ierr)
      if ( allocated(ptchgGrad))      deallocate(ptchgGrad, stat=ierr)
      CHECK_ERROR(ierr)
    enddo

    ! finalize QUICK, required only once
    call deleteQuickJob(ierr)
    CHECK_ERROR(ierr)

    ! deallocate memory
    if ( allocated(atomic_numbers)) deallocate(atomic_numbers, stat=ierr)
    if ( allocated(coord))          deallocate(coord, stat=ierr)
    if ( allocated(gradients))         deallocate(gradients, stat=ierr)
    CHECK_ERROR(ierr)

#ifdef MPIV
   call mpi_exit
#endif

  end program test_quick_api


