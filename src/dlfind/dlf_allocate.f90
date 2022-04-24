!!****h* utilities/dlf_allocate
!!
!! NAME
!! allocate
!!
!! FUNCTION
!! The allocation module
!!
!! This module should replace allocate and deallocate statements
!! it can only cover inctrinsic types, no derived types.
!!
!! it does compile and work with:
!!    g95
!!    ifort
!!    gfortran
!!
!! It does not work with:
!!    pgf95
!!
!! The preprocessor option OLDALLOC allows to fall back to traditional
!! f90 allcoation (to make the code run with pgf95). All calls to 
!! "call allocate(A,B)" are then replaced by an sed script (in the makefile)
!! by "allocate(A(B))"
!!
!! It allows for some bookkeeping of the storage, and a report of how much
!! memory had been used.
!!
!! In principle it can be used to define something like a high watermark
!! to make it impossible to allocate more memory than the watermark.
!!
!! DATA
!!  $Date$
!!  $Rev$
!!  $Author$
!!  $URL$
!!  $Id$
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!
!! SOURCE
module dlf_allocate

#ifdef OLDALLOC 

contains

  ! only dummies. These subroutines should never be called
  subroutine allocate_report
  end subroutine allocate_report
  subroutine allocate
  end subroutine allocate
  subroutine deallocate
  end subroutine deallocate

#else 

  use dlf_parameter_module, only: rk
  implicit none

  Public :: allocate
  Public :: deallocate
  Public :: allocate_report

  interface allocate
    module procedure allocate_r1
    module procedure allocate_r2
    module procedure allocate_r3
    module procedure allocate_i1
    module procedure allocate_i2
    module procedure allocate_l1
  end interface

  interface deallocate
    module procedure deallocate_r1
    module procedure deallocate_r2
    module procedure deallocate_r3
    module procedure deallocate_i1
    module procedure deallocate_i2
    module procedure deallocate_l1
  end interface

  private
  integer           :: fail
  logical,parameter :: tretry=.true. ! In case of an error, do the (de)allocate again.
        ! This will kill the program, but provide the error message from the compiler
  integer,save      :: stdout=2021
  integer,save      :: stderr=0
  logical,save      :: verbose=.false.
  integer,save      :: current(4)=0 ! real, integer, logical, complex
  integer,parameter :: a_kind(4)=(/kind(1.D0),kind(1),kind(.true.),2*kind((1.D0,1.D0)) /) 
  integer,save      :: storage=0
  integer,save      :: maxstorage=0

contains
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%                        Real Arrays                               %%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_allocate/allocate
!!
!! FUNCTION
!!
!! Allocate an array of type real or integer or logical.
!! In case of a real array, up to three dimensions are currently allowed,
!! two in integer arrays, and one in logical arrays
!!
!! SYNOPSIS
!  to be called by something like
!  call allocate (array,ubound1,ubound2)
!
!  the module then splits this to 
!  subroutine allocate_r1(array,ubound)
!  subroutine allocate_r2(array,ubound1,ubound2), ...
!
!! SOURCE
! ----------------------------------------------------------------------
  subroutine allocate_r1(array,ubound)
    real(rk),allocatable :: array(:)
    integer ,intent(in)  :: ubound

    if(verbose) write(stdout,"('Allocating real(:) array. Size:',i8)")&
        ubound

    allocate(array(ubound),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_r1')")
      if(tretry) allocate(array(ubound))
      call allocate_error(fail)
    end if

    call allocate_add(1,ubound)

  end subroutine allocate_r1

! ----------------------------------------------------------------------
  subroutine allocate_r2(array,ubound1,ubound2)
    real(rk),allocatable :: array(:,:)
    integer ,intent(in)  :: ubound1,ubound2

    if(verbose) write(stdout,"('Allocating real(:,:) array. Size:',i8)")&
        ubound1*ubound2

    allocate(array(ubound1,ubound2),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_r2')")
      if(tretry) allocate(array(ubound1,ubound2))
      call allocate_error(fail)
    end if

    call allocate_add(1,ubound1*ubound2)

  end subroutine allocate_r2

! ----------------------------------------------------------------------
  subroutine allocate_r3(array,ubound1,ubound2,ubound3)
    real(rk),allocatable :: array(:,:,:)
    integer ,intent(in)  :: ubound1,ubound2,ubound3

    if(verbose) write(stdout,"('Allocating real(:,:,:) array. Size:',i8)")&
        ubound1*ubound2*ubound3

    allocate(array(ubound1,ubound2,ubound3),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_r3')")
      if(tretry) allocate(array(ubound1,ubound2,ubound3))
      call allocate_error(fail)
    end if

    call allocate_add(1,ubound1*ubound2*ubound3)

  end subroutine allocate_r3
!!****

!!****f* dlf_allocate/deallocate
!!
!! FUNCTION
!!
!! Deallocate an array of type real or integer or logical.
!! In case of a real array, up to three dimensions are currently allowed,
!! two in integer arrays, and one in logical arrays
!!
!! SYNOPSIS
!  to be called by something like
!  call deallocate (array)
!
!  the module then splits this to 
!  subroutine deallocate_r1(array)
!  subroutine deallocate_r2(array), ...
!
!! SOURCE
! ----------------------------------------------------------------------
  subroutine deallocate_r1(array)
    real(rk),allocatable :: array(:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Dellocating real(:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_r1')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(1,ubound)

  end subroutine deallocate_r1

! ----------------------------------------------------------------------
  subroutine deallocate_r2(array)
    real(rk),allocatable :: array(:,:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Dellocating real(:,:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_r2')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(1,ubound)

  end subroutine deallocate_r2

! ----------------------------------------------------------------------
  subroutine deallocate_r3(array)
    real(rk),allocatable :: array(:,:,:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Dellocating real(:,:,:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_r3')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(1,ubound)

  end subroutine deallocate_r3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%                      Integer Arrays                              %%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! ----------------------------------------------------------------------
  subroutine allocate_i1(array,ubound)
    integer ,allocatable :: array(:)
    integer ,intent(in)  :: ubound

    if(verbose) write(stdout,"('Allocating integer(:) array. Size:',i8)")&
        ubound

    allocate(array(ubound),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_i1')")
      if(tretry) allocate(array(ubound))
      call allocate_error(fail)
    end if

    call allocate_add(2,ubound)

  end subroutine allocate_i1

! ----------------------------------------------------------------------
  subroutine allocate_i2(array,ubound1,ubound2)
    integer ,allocatable :: array(:,:)
    integer ,intent(in)  :: ubound1,ubound2

    if(verbose) write(stdout,"('Allocating integer(:,:) array. Size:',i8)")&
        ubound1*ubound2

    allocate(array(ubound1,ubound2),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_i2')")
      if(tretry) allocate(array(ubound1,ubound2))
      call allocate_error(fail)
    end if

    call allocate_add(2,ubound1*ubound2)

  end subroutine allocate_i2

! ----------------------------------------------------------------------
  subroutine deallocate_i1(array)
    integer ,allocatable :: array(:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Deallocating integer(:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_i1')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(2,ubound)

  end subroutine deallocate_i1

! ----------------------------------------------------------------------
  subroutine deallocate_i2(array)
    integer ,allocatable :: array(:,:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Deallocating integer(:,:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_i2')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(2,ubound)

  end subroutine deallocate_i2

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%                      Logical Arrays                              %%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! ----------------------------------------------------------------------
  subroutine allocate_l1(array,ubound)
    logical ,allocatable :: array(:)
    integer ,intent(in)  :: ubound

    if(verbose) write(stdout,"('Allocating logical(:) array. Size:',i8)")&
        ubound

    allocate(array(ubound),stat=fail)

    if(fail/=0) then
      write(stderr,"('Allocation error in allocate_l1')")
      if(tretry) allocate(array(ubound))
      call allocate_error(fail)
    end if

    call allocate_add(3,ubound)

  end subroutine allocate_l1

! ----------------------------------------------------------------------
  subroutine deallocate_l1(array)
    logical ,allocatable :: array(:)
    integer              :: ubound

    ubound=size(array)
    if(verbose) write(stdout,"('Deallocating logical(:) array. Size:',i8)")&
        ubound

    deallocate(array,stat=fail)

    if(fail/=0) then
      write(stderr,"('Deallocation error in deallocate_l1')")
      if(tretry) deallocate(array)
      call allocate_error(fail)
    end if

    call allocate_sub(3,ubound)

  end subroutine deallocate_l1

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine allocate_error(fail)
    integer, intent(in) :: fail
    print*,"Error number",fail
    ! g95: 211 - allocated
    !      210 - Operating system error: Cannot allocate memory
    ! ifort: 151 allocated
    !        179 too large Cannot allocate array - overflow on array size calculation.
    call dlf_mpi_abort()
    call exit(1)
  end subroutine allocate_error

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine allocate_add(type,a_size)
    integer,intent(in) :: type
    integer,intent(in) :: a_size

    if(type>4.or.type<1) then
       call dlf_mpi_abort()
       stop "Wrong type in allocate_add"
    end if
    current(type)=current(type)+a_size
    storage=storage + a_size * a_kind(type)
    if(storage>maxstorage) maxstorage=storage
    if(verbose) write(stdout,'("Current storage: ",i8," Max. Storage: ",&
        &i8)') storage,maxstorage
  end subroutine allocate_add

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine allocate_sub(type,a_size)
    integer,intent(in) :: type
    integer,intent(in) :: a_size

    if(type>4.or.type<1) then
       call dlf_mpi_abort()
       stop "Wrong type in allocate_sub"
    end if

    current(type)=current(type)-a_size
    storage=storage - a_size * a_kind(type)
    if(verbose) write(stdout,'("Current storage: ",i8," Max. Storage: ",&
        &i8)') storage,maxstorage
  end subroutine allocate_sub

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_allocate/allocate_report
!!
!! FUNCTION
!!
!! Report on maximum and currently allocated memory
!!
!! SYNOPSIS
  subroutine allocate_report
!! SOURCE
    if(storage/=0) then
      write(stdout,1000) "Current memory usage",dble(storage)/1024.D0,&
          "kB"
      write(stdout,"(a,i8)") "Currently allocated real values    ",current(1)
      write(stdout,"(a,i8)") "Currently allocated integer values ",current(2)
      write(stdout,"(a,i8)") "Currently allocated logical values ",current(3)
      write(stdout,"(a,i8)") "Currently allocated complex values ",current(4)
    end if
    write(stdout,1000) "Maximum memory usage",dble(maxstorage)/1024.D0,&
        "kB"
    call dlf_mpi_memory(storage,maxstorage)
    ! real number
1000 format (t1,'................................................', &
         t1,a,' ',t50,es10.4,1x,a)
  end subroutine allocate_report
!!****

#endif

end module dlf_allocate
