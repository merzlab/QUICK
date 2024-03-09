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

! Test module for QUICK library API
module test_quick_api_module

  implicit none
  private

  public :: loadTestData, printQuickOutput

#ifdef MPIV
  public :: mpi_initialize, printQuickMPIOutput, mpi_exit
#endif

  ! a test system with one water and 3 point charges
  ! atomic coordinates, external point charges and their coordinates
  ! for five snapshots 
  double precision, dimension(1:45) :: all_coords
  double precision, dimension(1:60) :: all_extchg

  data all_coords &
  /-0.778803, 0.000000, 1.132683, &
   -0.666682, 0.764099, 1.706291, &
   -0.666682,-0.764099, 1.706290, &
   -0.678803, 0.000008, 1.232683, &
   -0.724864, 0.755998, 1.606291, &
   -0.724862,-0.756005, 1.606290, &
   -0.714430, 0.000003, 1.267497, &
   -0.687724, 0.761169, 1.624424, &
   -0.687723,-0.761172, 1.624427, &
   -0.771504, 0.000000, 1.167497, &
   -0.669068, 0.763767, 1.697008, &
   -0.669068,-0.763767, 1.697008, &
   -0.771372, 0.000000, 1.162784, &
   -0.668845, 0.767538, 1.698983, &
   -0.668845,-0.767538, 1.698982/

    data all_extchg &
  /1.6492, 0.0000,-2.3560, -0.8340, &
   0.5448, 0.0000,-3.8000,  0.4170, &
   0.5448, 0.0000,-0.9121,  0.4170, &
   1.6492, 0.0000,-2.3560, -0.8360, &
   0.5448, 0.0000,-3.8000,  0.4160, &
   0.5448, 0.0000,-0.9121,  0.4160, &
   1.6492, 0.0000,-2.3560, -0.8380, &
   0.5448, 0.0000,-3.8000,  0.4150, &
   0.5448, 0.0000,-0.9121,  0.4150, &
   1.6492, 0.0000,-2.3560, -0.8400, &
   0.5448, 0.0000,-3.8000,  0.4140, &
   0.5448, 0.0000,-0.9121,  0.4140, &
   1.6492, 0.0000,-2.3560, -0.8420, &
   0.5448, 0.0000,-3.8000,  0.4130, &
   0.5448, 0.0000,-0.9121,  0.4130/

   ! number of point charges per frame
   integer :: nptg_pframe = 3

  interface loadTestData
    module procedure load_test_data
  end interface loadTestData

contains

  subroutine load_test_data(frame, natoms, nxt_charges, coord, xc_coord)

    implicit none

    integer, intent(in)             :: frame, natoms, nxt_charges
    double precision, intent(inout) :: coord(3, natoms)
    double precision, intent(out)   :: xc_coord(4, nxt_charges)
    integer :: i, j, k

    k=natoms*3*(frame-1) + 1
    do i=1,natoms
      do j=1,3
        coord(j,i) = all_coords(k)
        k=k+1
      enddo
    enddo

    if(nxt_charges>0) then
      k=nptg_pframe*4*(frame-1) + 1
      do i=1,nxt_charges
        do j=1,4
          xc_coord(j,i) = all_extchg(k)
          k=k+1
        enddo
      enddo
    endif

  end subroutine load_test_data

#ifdef MPIV
  ! initialize mpi library and save mpirank and mpisize
  subroutine mpi_initialize(mpisize, mpirank, master, mpierror)

    use mpi
    implicit none

    integer, intent(inout) :: mpisize, mpirank, mpierror
    logical, intent(inout) :: master

    call MPI_INIT(mpierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

    if(mpirank .eq. 0) then
      master = .true.
    else
      master = .false.
    endif

  end subroutine mpi_initialize

  ! prints mpi output sequentially
  subroutine printQuickMPIOutput(natoms, nxt_charges, atomic_numbers, totEne, gradients, ptchg_grad, mpirank)

    implicit none

    integer, intent(in)          :: natoms, nxt_charges, mpirank
    integer, intent(in)          :: atomic_numbers(natoms)
    double precision, intent(in) :: totEne
    double precision, intent(in) :: gradients(3,natoms)
    double precision, intent(in) :: ptchg_grad(3,nxt_charges)

    write(*,*) ""
    write(*,'(A11, 1X, I3, 1x, A3)') "--- MPIRANK", mpirank, "---"
    write(*,*) ""

    call printQuickOutput(natoms, nxt_charges, atomic_numbers, totEne, gradients, ptchg_grad)

  end subroutine printQuickMPIOutput

  subroutine mpi_exit

    use mpi
    implicit none
    integer :: mpierror

    call MPI_FINALIZE(mpierror)
    call exit(0)

  end subroutine mpi_exit

#endif


  subroutine printQuickOutput(natoms, nxt_charges, atomic_numbers, totEne, gradients, ptchg_grad)

    implicit none

    integer, intent(in)          :: natoms, nxt_charges
    integer, intent(in)          :: atomic_numbers(natoms)
    double precision, intent(in) :: totEne
    double precision, intent(in) :: gradients(3,natoms)
    double precision, intent(in) :: ptchg_grad(3,nxt_charges)
    integer :: i, j

    ! print energy  
    write(*,*) ""
    write(*,*) "*** TESTING QUICK API ***"
    write(*,*) ""
    write(*,*) "PRINTING ENERGY"
    write(*,*) "---------------"
    write(*,*) ""
    write(*, '(A14, 3x, F14.10, 1x, A4)') "TOTAL ENERGY =",totEne,"A.U."

    ! print gradients
    write(*,*) ""
    write(*,*) "PRINTING GRADIENTS"
    write(*,*) "------------------"
    write(*,*) ""
    write(*, '(A14, 3x, A6, 10x, A6, 10x, A6)') "ATOMIC NUMBER","GRAD-X","GRAD-Y","GRAD-Z"

    do i=1,natoms
      write(*,'(6x, I5, 2x, F14.10, 2x, F14.10, 2x, F14.10)') atomic_numbers(i), gradients(1,i), gradients(2,i), gradients(3,i)
    enddo

    ! print point charge gradients
    if(nxt_charges>0) then
      write(*,*) ""
      write(*,*) "PRINTING POINT CHARGE GRADIENTS"
      write(*,*) "-------------------------------"
      write(*,*) ""
      write(*, '(A14, 3x, A6, 10x, A6, 10x, A6)') "CHARGE NUMBER","GRAD-X","GRAD-Y","GRAD-Z"

      do i=1,nxt_charges
        write(*,'(6x, I5, 2x, F14.10, 2x, F14.10, 2x, F14.10)') i, ptchg_grad(1,i), ptchg_grad(2,i), ptchg_grad(3,i)
      enddo
    endif

    write(*,*) ""

  end subroutine printQuickOutput

end module
