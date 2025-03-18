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

#include "util.fh"

module quick_restart_module

  use HDF5
  use ISO_C_BINDING

  use quick_files_module, only: dataFileName
  use quick_exception_module, only: RaiseException

  implicit none

  public :: data_write_info, write_integer_array, write_double_array
  public :: iread, aread

  interface write_double_array
    module procedure write_double_2d_array
  end interface write_double_array

  interface aread
    module procedure aread_point
    module procedure aread_point_2d
    module procedure areadn
    module procedure areadn_2d
  end interface aread

  interface iread
    module procedure iread_point
    module procedure iread_point_2d
    module procedure ireadn
    module procedure ireadn_2d
  end interface iread

contains

  subroutine aread_point(datasetname, ind, adata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 1
    INTEGER, intent(in)             :: ind
    double precision, intent(out)   :: adata
    double precision                :: abuf(1,1)

    INTEGER :: hdferr

    INTEGER(HID_T)                :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)              :: dims(1,1)
    INTEGER(HSIZE_T), parameter   :: dimn(2) = (/1,1/)
    INTEGER(HSIZE_T), parameter   :: npoints = 1
    !
    ! index at which the data is located
    !
    dims(1,1) = ind
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a scalar datapsace
    !
    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)
    !
    ! Read the array in dataset "datasetname"
    !
    call h5dread_f(dset, H5T_NATIVE_INTEGER, abuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    adata = abuf(1,1)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine aread_point

  subroutine aread_point_2d(datasetname, ind, adata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 2
    INTEGER, intent(in)             :: ind(2)
    double precision, intent(out)   :: adata
    double precision                :: abuf(1,1)

    INTEGER :: hdferr

    INTEGER(HID_T)                :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)              :: dims(1,2)
    INTEGER(HSIZE_T), parameter   :: dimn(2)=(/1,1/)
    INTEGER(HSIZE_T), parameter   :: npoints = 1
    !
    ! index at which the data is located
    !
    dims(1,:)=(/ind(1),ind(2)/)
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a scalar datapsace
    !
    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)
    !
    ! Read the array in dataset "datasetname"
    !
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, abuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    adata = abuf(1,1)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine aread_point_2d

  subroutine areadn(datasetname, ind, n, adata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 1
    INTEGER, intent(in)             :: ind, n
    double precision, intent(out)   :: adata(:)
    double precision, allocatable   :: abuf(:)

    INTEGER :: hdferr

    INTEGER(HID_T)      :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)    :: dims(1)
    INTEGER(HSIZE_T)    :: start(1), stride(1), countn(1), blockn(1)
    !
    ! index at which the data is located
    !
    dims(1) = n
    start  = [ind-1]
    stride = [1]
    countn = [1]
    blockn = [n]
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a simple datapsace
    !
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Read the array in dataset "datasetname"
    !
    allocate(abuf(n))
    call h5dread_f(dset, H5T_NATIVE_INTEGER, abuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    adata(:) = abuf(:)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine areadn

  subroutine areadn_2d(datasetname, ind, n, adata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 2
    INTEGER, intent(in)             :: ind(2), n(2)
    double precision, intent(out)   :: adata(:,:)
    double precision, allocatable   :: abuf(:,:)

    INTEGER :: hdferr

    INTEGER(HID_T)      :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)    :: dims(2)
    INTEGER(HSIZE_T)    :: start(2), stride(2), countn(2), blockn(2)
    !
    ! index at which the data is located
    !
    dims(:) = n(:)
    start  = [ind(1)-1,ind(2)-1]
    stride = [1,1]
    countn = [1,1]
    blockn = [n(1),n(2)]
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a simple datapsace
    !
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Read the array in dataset "datasetname"
    !
    allocate(abuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, abuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    adata(:,:) = abuf(:,:)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine areadn_2d

  subroutine iread_point_2d(datasetname, ind, idata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 2
    INTEGER, intent(in)             :: ind(2)
    integer, intent(out)            :: idata
    integer                         :: ibuf(1,1)

    INTEGER :: hdferr

    INTEGER(HID_T)                :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)              :: dims(1,2)
    INTEGER(HSIZE_T), parameter   :: dimn(2)=(/1,1/)
    INTEGER(HSIZE_T), parameter   :: npoints = 1
    !
    ! index at which the data is located
    !
    dims(1,:)=(/ind(1),ind(2)/)
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a scalar datapsace
    !
    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)
    !
    ! Read the array in dataset "datasetname"
    !
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    idata = ibuf(1,1)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine iread_point_2d

  subroutine iread_point(datasetname, ind, idata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 1
    INTEGER, intent(in)             :: ind
    INTEGER, intent(out)            :: idata
    INTEGER                         :: ibuf(1,1)

    INTEGER :: hdferr

    INTEGER(HID_T)                :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)              :: dims(1,1)
    INTEGER(HSIZE_T), parameter   :: dimn(2) = (/1,1/)
    INTEGER(HSIZE_T), parameter   :: npoints = 1
    !
    ! index at which the data is located
    !
    dims(1,1) = ind
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a scalar datapsace
    !
    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)
    !
    ! Read the array in dataset "datasetname"
    !
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    idata = ibuf(1,1)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine iread_point

  subroutine ireadn(datasetname, ind, n, idata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 1
    INTEGER, intent(in)             :: ind, n
    INTEGER, intent(out)            :: idata(:)
    INTEGER, allocatable            :: ibuf(:)

    INTEGER :: hdferr

    INTEGER(HID_T)      :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)    :: dims(1)
    INTEGER(HSIZE_T)    :: start(1), stride(1), countn(1), blockn(1)
    !
    ! index at which the data is located
    !
    dims(1) = n
    start  = [ind-1]
    stride = [1]
    countn = [1]
    blockn = [n]
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a simple datapsace
    !
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Read the array in dataset "datasetname"
    !
    allocate(ibuf(n))
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    idata(:) = ibuf(:)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine ireadn

  subroutine ireadn_2d(datasetname, ind, n, idata)

    implicit none

    CHARACTER(LEN=*), intent(in)    :: datasetname
    INTEGER, parameter              :: rank = 2
    INTEGER, intent(in)             :: ind(2), n(2)
    INTEGER, intent(out)            :: idata(:,:)
    INTEGER, allocatable            :: ibuf(:,:)

    INTEGER :: hdferr

    INTEGER(HID_T)      :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T)    :: dims(2)
    INTEGER(HSIZE_T)    :: start(2), stride(2), countn(2), blockn(2)
    !
    ! index at which the data is located
    !
    dims(:) = n(:)
    start  = [ind(1)-1,ind(2)-1]
    stride = [1,1]
    countn = [1,1]
    blockn = [n(1),n(2)]
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! open dataset
    !
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Get the file_space
    !
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! create a simple datapsace
    !
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Read the array in dataset "datasetname"
    !
    allocate(ibuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! The data in the buffer is transferred to return to the calling program
    !
    idata(:,:) = ibuf(:,:)
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine ireadn_2d

  subroutine data_write_info()

    use quick_molspec_module, only: natom, quick_molspec

    implicit none

    CHARACTER(LEN=7) , PARAMETER :: datasetname = "molinfo"
    INTEGER          , PARAMETER :: dim0       = 2

    INTEGER, DIMENSION(1:dim0) :: wdata

    INTEGER            :: hdferr
    INTEGER, PARAMETER :: rank = 1
    INTEGER(HSIZE_T)   :: length(rank)

    INTEGER(HID_T)  :: file, space_id, dset ! Handles
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)

    wdata = (/natom,quick_molspec%nbasis/)
    length = shape(wdata)
    !
    ! Create a new file using the default properties.
    !
    CALL h5fcreate_f(dataFileName, H5F_ACC_TRUNC_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Create a simple dataspace
    !
    call H5Screate_simple_f(rank, length, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Create dataset
    !
    call h5dcreate_f(file, datasetname, H5T_NATIVE_INTEGER, space_id, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Write the array to a dataset "datasetname"
    !
    call h5dwrite_f(dset, H5T_NATIVE_INTEGER, wdata, length, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine data_write_info

  subroutine write_integer_array(Array, length, datasetname)

    implicit none

    CHARACTER(LEN=*)                 :: datasetname
    INTEGER, PARAMETER               :: rank = 1
    INTEGER                          :: length
    INTEGER(HSIZE_T)                 :: lenArr(rank)
    INTEGER, DIMENSION(length)       :: Array

    INTEGER :: hdferr
    INTEGER(HID_T)  :: file, space_id, dset ! Handles  
    !
    lenArr=shape(Array)
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file.
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Create a simple dataspace
    !
    call H5Screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Create dataset
    !
    call h5dcreate_f(file, datasetname, H5T_NATIVE_INTEGER, space_id, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Write the array to a dataset "datasetname"
    !
    call h5dwrite_f(dset, H5T_NATIVE_INTEGER, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine write_integer_array

  subroutine write_double_2d_array(Array, length1, length2, datasetname)

    implicit none

    CHARACTER(LEN=*)                                 :: datasetname
    INTEGER, PARAMETER                               :: rank = 2
    INTEGER                                          :: length1, length2
    INTEGER(HSIZE_T)                                 :: lenArr(rank)
    double precision, DIMENSION(length1, length2)    :: Array

    INTEGER :: hdferr
    logical :: exists
    INTEGER(HID_T)  :: file, space_id, dset ! Handles  
    !
    lenArr=shape(Array)
    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(hdferr)
    !
    ! Open file.
    !
    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Before writing check if dataset exists
    !
    call h5lexists_f(file, datasetname, exists, hdferr)
    !
    ! delete the dataset if exists
    !
    if (exists) then
      call h5ldelete_f(file, datasetname, hdferr)
    endif
    !
    ! Create a simple dataspace
    !
    call H5Screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Create dataset
    !
    call h5dcreate_f(file, datasetname, H5T_NATIVE_DOUBLE, space_id, dset, hdferr)
    if (hdferr /= 0)then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Write the array to a dataset "datasetname"
    !
    call h5dwrite_f(dset, H5T_NATIVE_DOUBLE, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif
    !
    ! Close file and dataset
    !
    call h5sclose_f(space_id, hdferr)
    call h5dclose_f(dset, hdferr)
    call h5fclose_f(file, hdferr)

  end subroutine write_double_2d_array

end module quick_restart_module
