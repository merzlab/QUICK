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

#if defined(RESTART_HDF5)

  implicit none

  interface read_hdf5_double_2n
    module procedure read_double_2n
  end interface read_hdf5_double_2n

  interface read_hdf5_int
    module procedure read_int
  end interface read_hdf5_int

  interface read_hdf5_int_n
    module procedure read_int_n
  end interface read_hdf5_int_n

  interface read_hdf5_int_2n
    module procedure read_int_2n
  end interface read_hdf5_int_2n

  interface write_hdf5_info
    module procedure write_info
  end interface write_hdf5_info

  interface write_hdf5_int_n
    module procedure write_int_n
  end interface write_hdf5_int_n

  interface write_hdf5_double_2n
    module procedure write_double_2n
  end interface write_hdf5_double_2n

contains

  subroutine read_double_2n(datasetname, ind, n, adata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    CHARACTER(LEN=*), intent(in) :: datasetname
    INTEGER, intent(in) :: ind(2)
    INTEGER, intent(in) :: n(2)
    double precision, intent(out) :: adata(:,:)

    INTEGER, parameter :: rank = 2
    double precision, allocatable :: abuf(:,:)
    INTEGER :: hdferr
    INTEGER(HID_T) :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T) :: dims(2)
    INTEGER(HSIZE_T) :: start(2), stride(2), countn(2), blockn(2)

    ! index at which the data is located
    dims(:) = n(:)
    start  = [ind(1)-1,ind(2)-1]
    stride = [1,1]
    countn = [1,1]
    blockn = [n(1),n(2)]

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! open dataset
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! create a simple datapsace
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(abuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, abuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    adata(:,:) = abuf(:,:)

    if(allocated(abuf)) deallocate(abuf)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_double_2n

  subroutine read_int(datasetname, ind, idata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    CHARACTER(LEN=*), intent(in) :: datasetname
    INTEGER, intent(in) :: ind
    INTEGER, intent(out) :: idata

    INTEGER, parameter :: rank = 1
    INTEGER :: ibuf(1,1)
    INTEGER :: hdferr
    INTEGER(HID_T) :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T) :: dims(1,1)
    INTEGER(HSIZE_T), parameter :: dimn(2) = (/1,1/)
    INTEGER(HSIZE_T), parameter :: npoints = 1

    ! index at which the data is located
    dims(1,1) = ind

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! open dataset
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! create a scalar datapsace
    call H5Screate_F(H5S_SCALAR_F, space_id, hdferr)

    ! Read the array in dataset "datasetname"
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata = ibuf(1,1)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_int

  subroutine read_int_n(datasetname, ind, n, idata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    CHARACTER(LEN=*), intent(in) :: datasetname
    INTEGER, intent(in) :: ind
    INTEGER, intent(in) :: n
    INTEGER, intent(out) :: idata(:)

    INTEGER, parameter :: rank = 1
    INTEGER, allocatable :: ibuf(:)
    INTEGER :: hdferr
    INTEGER(HID_T) :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T) :: dims(1)
    INTEGER(HSIZE_T) :: start(1), stride(1), countn(1), blockn(1)

    ! index at which the data is located
    dims(1) = n
    start  = [ind-1]
    stride = [1]
    countn = [1]
    blockn = [n]

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! open dataset
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! create a simple datapsace
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(ibuf(n))
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata(:) = ibuf(:)

    if(allocated(ibuf)) deallocate(ibuf)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_int_n

  subroutine read_int_2n(datasetname, ind, n, idata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    CHARACTER(LEN=*), intent(in) :: datasetname
    INTEGER, intent(in) :: ind(2)
    INTEGER, intent(in) :: n(2)
    INTEGER, intent(out) :: idata(:,:)

    INTEGER, parameter :: rank = 2
    INTEGER, allocatable :: ibuf(:,:)
    INTEGER :: hdferr
    INTEGER(HID_T) :: file, dset, space_id, file_space ! Handles
    INTEGER(HSIZE_T) :: dims(2)
    INTEGER(HSIZE_T) :: start(2), stride(2), countn(2), blockn(2)

    ! index at which the data is located
    dims(:) = n(:)
    start  = [ind(1)-1,ind(2)-1]
    stride = [1,1]
    countn = [1,1]
    blockn = [n(1),n(2)]

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    CALL h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! open dataset
    call h5dopen_f(file, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error getting space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! create a simple datapsace
    call H5Screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(ibuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_INTEGER, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error reading data from dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata(:,:) = ibuf(:,:)

    if(allocated(ibuf)) deallocate(ibuf)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_int_2n

  subroutine write_info(natom, nbasis)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    INTEGER, intent(in) :: natom
    INTEGER, intent(in) :: nbasis

    CHARACTER(LEN=7), PARAMETER :: datasetname = "molinfo"
    INTEGER, PARAMETER :: dim0 = 2
    INTEGER, DIMENSION(1:dim0) :: wdata
    INTEGER :: hdferr
    INTEGER, PARAMETER :: rank = 1
    INTEGER(HSIZE_T) :: length(rank)
    INTEGER(HID_T) :: file, space_id, dset ! Handles

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    wdata = (/natom,nbasis/)
    length = shape(wdata)

    ! Create a new file using the default properties.
    CALL h5fcreate_f(dataFileName, H5F_ACC_TRUNC_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 data file')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call H5Screate_simple_f(rank, length, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file, datasetname, H5T_NATIVE_INTEGER, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_INTEGER, wdata, length, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_info

  subroutine write_int_n(Array, length, datasetname)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    INTEGER, intent(in) :: length
    INTEGER, DIMENSION(length), intent(in) :: Array
    CHARACTER(LEN=*), intent(in) :: datasetname

    INTEGER, PARAMETER :: rank = 1
    INTEGER(HSIZE_T) :: lenArr(rank)

    INTEGER :: hdferr
    INTEGER(HID_T)  :: file, space_id, dset ! Handles  

    lenArr=shape(Array)

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Create a simple dataspace
    call H5Screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Create dataset
    call h5dcreate_f(file, datasetname, H5T_NATIVE_INTEGER, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_INTEGER, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_int_n

  subroutine write_double_2n(Array, length1, length2, datasetname)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    INTEGER, intent(in) :: length1, length2
    double precision, DIMENSION(length1, length2), intent(in) :: Array
    CHARACTER(LEN=*), intent(in) :: datasetname

    INTEGER, PARAMETER :: rank = 2
    INTEGER(HSIZE_T) :: lenArr(rank)
    INTEGER :: hdferr
    logical :: exists
    INTEGER(HID_T) :: file, space_id, dset ! Handles  

    lenArr=shape(Array)

    ! Initialize FORTRAN interface.
    CALL h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    CALL h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to open HDF5 data file')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Before writing check if dataset exists
    call h5lexists_f(file, datasetname, exists, hdferr)

    ! delete the dataset if exists
    if (exists) then
      call h5ldelete_f(file, datasetname, hdferr)
    endif

    ! Create a simple dataspace
    call H5Screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error creating space in the HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Create dataset
    call h5dcreate_f(file, datasetname, H5T_NATIVE_DOUBLE, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Failed to create HDF5 dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_DOUBLE, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE,'Error writing data to dataset')
      call quick_exit(OUTFILEHANDLE,1)
    endif

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_double_2n

#endif

end module quick_restart_module
