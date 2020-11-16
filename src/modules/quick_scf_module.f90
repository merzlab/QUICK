!---------------------------------------------------------------------!
! Created by Madu Manathunga on 06/24/2020                            !
!                                                                     ! 
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

! This module contains subroutines and data structures related to scf
! and diis procedures

module quick_scf_module

  implicit none
  private

  public :: allocate_quick_scf, deallocate_quick_scf 
  public :: V2, oneElecO, B, BSAVE, BCOPY, W, COEFF, RHS, allerror, alloperator
!  type quick_scf_type

    ! a workspace matrix of size 3,nbasis to be passed into the diagonalizer 
    double precision, allocatable, dimension(:,:) :: V2

    ! one electron operator, a matrix of size nbasis,nbasis
    double precision, allocatable, dimension(:,:)   :: oneElecO

    ! matrices required for diis procedure
    double precision, allocatable, dimension(:,:)   :: B

    double precision, allocatable, dimension(:,:)   :: BSAVE

    double precision, allocatable, dimension(:,:)   :: BCOPY

    double precision, allocatable, dimension(:)     :: W

    double precision, allocatable, dimension(:)     :: COEFF

    double precision, allocatable, dimension(:)     :: RHS

    double precision, allocatable, dimension(:,:,:) :: allerror

    double precision, allocatable, dimension(:,:,:) :: alloperator

!  end type quick_scf_type

!  type (quick_scf_type), save :: quick_scf

contains

! This subroutine allocates memory for quick_scf type and initializes them to zero. 
  subroutine allocate_quick_scf()

    use quick_method_module
    use quick_basis_module

    implicit none 

    integer :: ierr

    if(.not. allocated(V2))          allocate(V2(3, nbasis), stat=ierr)
    if(.not. allocated(oneElecO))    allocate(oneElecO(nbasis, nbasis), stat=ierr)
    if(.not. allocated(B))           allocate(B(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(BSAVE))       allocate(BSAVE(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(BCOPY))       allocate(BCOPY(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(W))           allocate(W(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(COEFF))       allocate(COEFF(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(RHS))         allocate(RHS(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(allerror))    allocate(allerror(nbasis, nbasis, quick_method%maxdiisscf), stat=ierr)
    if(.not. allocated(alloperator)) allocate(alloperator(nbasis, nbasis, quick_method%maxdiisscf), stat=ierr)

    !initialize values to zero
    V2          = 0.0d0
    oneElecO    = 0.0d0
    B           = 0.0d0
    BSAVE       = 0.0d0
    BCOPY       = 0.0d0
    W           = 0.0d0
    COEFF       = 0.0d0
    RHS         = 0.0d0
    allerror    = 0.0d0
    alloperator = 0.0d0

  end subroutine allocate_quick_scf 

  subroutine deallocate_quick_scf()

    implicit none

    integer :: ierr

    if(allocated(V2))          deallocate(V2, stat=ierr)
    if(allocated(oneElecO))    deallocate(oneElecO, stat=ierr)
    if(allocated(B))           deallocate(B, stat=ierr)
    if(allocated(BSAVE))       deallocate(BSAVE, stat=ierr)
    if(allocated(BCOPY))       deallocate(BCOPY, stat=ierr)
    if(allocated(W))           deallocate(W, stat=ierr)
    if(allocated(COEFF))       deallocate(COEFF, stat=ierr)
    if(allocated(RHS))         deallocate(RHS, stat=ierr)
    if(allocated(allerror))    deallocate(allerror, stat=ierr)
    if(allocated(alloperator)) deallocate(alloperator, stat=ierr)

  end subroutine deallocate_quick_scf

end module quick_scf_module
