!---------------------------------------------------------------------!
! Created by Vikrant Tripathy on 03/07/2025                           !
!                                                                     !
! Copyright (C) 2024-2025 Merz lab                                    !
! Copyright (C) 2024-2025 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module manages the data file enabling restarting of quick      !
! calculation from density and/or coordinates.                        !
!---------------------------------------------------------------------!

module quick_restart_module

  implicit none

  public :: data_read, data_write

contains

  subroutine data_write_info()

  implicit none

  use quick_files_module, only: dataFileName
  use quick_molspec_module, only: natom, quick_molspec

  USE hdf5_utils
  USE HDF5
  USE ISO_C_BINDING

  CHARACTER(LEN=7) , PARAMETER :: attribute  = "molinfo"
  INTEGER          , PARAMETER :: dim0       = 2

  INTEGER, DIMENSION(1:dim0) :: wdata

  INTEGER :: hdferr

  INTEGER(HID_T)  :: file ! Handles
  !
  ! Initialize FORTRAN interface.
  !
  CALL h5open_f(hdferr)

  wdata = (/natom,quick_molspec%nbasis/)
  !
  ! Create a new file using the default properties.
  !
  CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file, hdferr)

  call write_to_hdf5(wdata, attribute, file, hdferr) 
  !
  ! Close file
  !
  call close_hdf5file(file, hdferr)

  end subroutine data_write_info

!  type restart_index_type
!
!    ! number of atoms
!    integer,dimension(2) :: natom
!
!    ! number of basis functions
!    integer,dimension(2) :: nbasis
!
!    ! density matrix
!    integer,dimension(2) :: dense
!
!    ! beta density matrix
!    integer,dimension(2) :: denseb
!
!    ! which atom type id every atom crosponds to
!    integer,dimension(2) :: iattype
!
!    ! Most recent xyz
!    integer,dimension(2) :: xyz
!
!    ! all the xyz coordinates including the intermediate
!    ! steps during optimization calculations
!    integer,dimension(2) :: all_xyz
!
!  end type restart_index_type
!
!  type(restart_index_type),save :: restart_index
!
!  subroutine init_restart_index(self)
!
!    implicit none
!    type(restart_index_type) :: self
!
!    self%natom   = 0
!    self%nbasis  = 0
!    self%dense   = 0
!    self%denseb  = 0
!    self%iattype = 0
!    self%xyz     = 0
!    self%all_xyz = 0
!
!  end subroutine init_restart_index

end module quick_restart_module
