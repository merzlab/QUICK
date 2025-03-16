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

!#include "util.fh"

module quick_restart_module

  use hdf5_utils
  use HDF5
  use ISO_C_BINDING

  use quick_files_module, only: dataFileName
!  use quick_exception_module, only: RaiseException

  implicit none

!  public :: data_write_info, write_integer_array, write_double_array
!  public :: iread

!  interface write_double_array
!    module procedure write_double_2d_array
!  end interface write_double_array

contains

!  subroutine iread(datasetname, ind, idata)
!
!    implicit none
!
!    CHARACTER(LEN=*)        :: datasetname
!    INTEGER                 :: idata, ind, rank
!
!    INTEGER :: hdferr
!
!    INTEGER(HID_T)   :: file, dset, space_id, file_space ! Handles
!    INTEGER(HSIZE_T) :: dims(1), npoints
!    !
!    dims(1) = ind
!    !
!    ! Initialize FORTRAN interface.
!    !
!    CALL h5open_f(hdferr)
!    !
!    ! Open file
!    !
!    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
!    if (hdferr /= 0)then
!      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! open dataset
!    !
!    call h5dopen_f(file, datasetname, dset, hdferr)
!    if (hdferr /= 0)then
!      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! Get the file_space
!    !
!    call h5dget_space_f(dset, file_space, hdferr)
!    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
!    if (hdferr /= 0) then
!      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! create a scalar datapsace
!    !
!    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)
!    !
!    ! Read the array in dataset "datasetname" from  ind(1) to icount(1)
!    !
!    call h5dread_f(dset, H5T_NATIVE_INTEGER, idata, dims, hdferr, file_space_id=file_space, mem_space_id = space_id)
!    if (hdferr /= 0) then
!      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! Close file and dataset
!    !
!    call h5sclose_f(space_id, hdferr)
!    call h5dclose_f(dset, hdferr)
!    call h5fclose_f(file, hdferr)
!
!  end subroutine iread

!  subroutine iread_1d(datasetname, ind, icount, idata)
!
!    implicit none
!
!    CHARACTER(LEN=*)        :: datasetname
!    INTEGER, allocatable    :: idata(:)
!
!    INTEGER :: hdferr
!
!    INTEGER(HID_T)   :: file, dset ! Handles
!    INTEGER(HSIZE_T) :: dims, ind, icount
!    !
!    ! Initialize FORTRAN interface.
!    !
!    CALL h5open_f(hdferr)
!    !
!    ! Open file
!    !
!    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
!    if (hdferr /= 0)then
!      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! open dataset
!    !
!    call h5dopen_f(file, datasetname, dset, hdferr)
!    if (hdferr /= 0)then
!      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! Get the dimensions of the dataset
!    !
!    call h5dget_space_f(dset, dims, hdferr)
!    if (hdferr /= 0) then
!      call PrtErr(OUTFILEHANDLE,'Error getting dataset dimension')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! Allocate the space to data to be read
!    !
!    allocate(idata(icount))
!    !
!    ! Read the array in dataset "datasetname" from  ind(1) to icount(1)
!    !
!    call h5dread_f(dset, H5T_NATIVE_INTEGER, idata, ind, icount, hdferr)
!    if (hdferr /= 0) then
!      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
!      call quick_exit(OUTFILEHANDLE,1)
!    endif
!    !
!    ! Close file and dataset
!    !
!    call h5dclose_f(dset, hdferr)
!    call h5fclose_f(file, hdferr)
!
!  end subroutine iread_1d

!  subroutine data_write_info()
!
!    use quick_molspec_module, only: natom, quick_molspec
!
!    implicit none
!
!    CHARACTER(LEN=7) , PARAMETER :: attribute  = "molinfo"
!    INTEGER          , PARAMETER :: dim0       = 2
!
!    INTEGER, DIMENSION(1:dim0) :: wdata
!
!    INTEGER :: hdferr
!
!    INTEGER(HID_T)  :: file ! Handles
!    !
!    ! Initialize FORTRAN interface.
!    !
!    CALL h5open_f(hdferr)
!
!    wdata = (/natom,quick_molspec%nbasis/)
!
!    !
!    ! Create a new file using the default properties.
!    !
!    CALL h5fcreate_f(dataFileName, H5F_ACC_TRUNC_F, file, hdferr)
!
!    call write_to_hdf5(wdata, attribute, file, hdferr) 
!    !
!    ! Close file
!    !
!    call close_hdf5file(file, hdferr)
!
!  end subroutine data_write_info
!
!  subroutine write_integer_array(Array, length, attribute)
!
!    implicit none
!
!    CHARACTER(LEN=*)           :: attribute
!    INTEGER                    :: length
!    INTEGER, DIMENSION(length) :: Array
!
!    INTEGER :: hdferr
!    INTEGER(HID_T)  :: file ! Handles  
!    !
!    ! Initialize FORTRAN interface.
!    !
!    CALL h5open_f(hdferr)
!    !
!    ! Open file.
!    !
!    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
!    !
!    ! Write the integer array as a dataset to the data file.
!    !
!    call write_to_hdf5(Array, attribute, file, hdferr)
!    !
!    ! Close file
!    !
!    call close_hdf5file(file, hdferr)
!  
!  end subroutine write_integer_array
!
!  subroutine write_double_2d_array(Array, length1, length2, attribute)
!
!    implicit none
!
!    CHARACTER(LEN=*)                                 :: attribute
!    INTEGER                                          :: length1, length2
!    double precision, DIMENSION(length1, length2)    :: Array
!
!    INTEGER :: hdferr
!    logical :: exists
!    INTEGER(HID_T)  :: file ! Handles  
!    !
!    ! Initialize FORTRAN interface.
!    !
!    CALL h5open_f(hdferr)
!    !
!    ! Open file.
!    !
!    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
!    !
!    ! Before writing check if dataset exists
!    !
!    call h5lexists_f(file, attribute, exists, hdferr)
!    !
!    ! delete the dataset if exists
!    !
!    if (exists) then
!      call h5ldelete_f(file, attribute, hdferr)
!    endif
!    !
!    ! Write the integer array as a dataset to the data file.
!    !
!    call write_to_hdf5(Array, attribute, file, hdferr)
!    !
!    ! Close file
!    !
!    call close_hdf5file(file, hdferr)
! 
!  end subroutine write_double_2d_array

end module quick_restart_module
