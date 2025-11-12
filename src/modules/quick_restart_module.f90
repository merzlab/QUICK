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

    character(len=*), intent(in) :: datasetname
    integer, dimension(2), intent(in) :: ind
    integer, dimension(2), intent(in) :: n
    double precision, dimension(n(1),n(2)), intent(out) :: adata

    integer, parameter :: rank = 2
    double precision, dimension(:,:), allocatable :: abuf
    integer :: hdferr
    integer(HID_T) :: file_id, dset, space_id, file_space ! Handles
    integer(HSIZE_T), dimension(2) :: dims
    integer(HSIZE_T), dimension(2) :: start, stride, countn, blockn

    ! index at which the data is located
    dims = n
    start  = ind - 1
    stride = [1, 1]
    countn = [1, 1]
    blockn = n

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a simple datapsace
    call h5screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(abuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, abuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    adata = abuf

    if(allocated(abuf)) deallocate(abuf)

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_double_2n

  subroutine read_int(datasetname, ind, idata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    character(len=*), intent(in) :: datasetname
    integer, intent(in) :: ind
    integer, intent(out) :: idata

    integer, parameter :: rank = 1
    integer, dimension(1,1) :: ibuf
    integer :: hdferr
    integer(HID_T) :: file_id, dset, space_id, file_space ! Handles
    integer(HSIZE_T), dimension(1,1) :: dims
    integer(HSIZE_T), dimension(2), parameter :: dimn = (/1,1/)
    integer(HSIZE_T), parameter :: npoints = 1

    ! index at which the data is located
    dims(1,1) = ind

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a scalar datapsace
    call h5screate_f(H5S_SCALAR_F, space_id, hdferr)

    ! Read the array in dataset "datasetname"
    call h5dread_f(dset, H5T_NATIVE_integer, ibuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata = ibuf(1,1)

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_int)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_int

  subroutine read_int_n(datasetname, ind, n, idata)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    character(len=*), intent(in) :: datasetname
    integer, intent(in) :: ind
    integer, intent(in) :: n
    integer, dimension(n), intent(out) :: idata

    integer, parameter :: rank = 1
    integer, dimension(:), allocatable :: ibuf
    integer :: hdferr
    integer(HID_T) :: file_id, dset, space_id, file_space ! Handles
    integer(HSIZE_T), dimension(1) :: dims
    integer(HSIZE_T), dimension(1) :: start, stride, countn, blockn

    ! index at which the data is located
    dims(1) = n
    start  = [ind - 1]
    stride = [1]
    countn = [1]
    blockn = [n]

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a simple datapsace
    call h5screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(ibuf(n))
    call h5dread_f(dset, H5T_NATIVE_integer, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata = ibuf

    if(allocated(ibuf)) deallocate(ibuf)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_int_n

  subroutine write_info(natom, nbasis)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    integer, intent(in) :: natom
    integer, intent(in) :: nbasis

    character(len=7), parameter :: datasetname = "molinfo"
    integer, dimension(2) :: wdata
    integer :: hdferr
    integer, parameter :: rank = 1
    integer(HSIZE_T), dimension(rank) :: length
    integer(HID_T) :: file_id, space_id, dset ! Handles

    wdata = (/natom, nbasis/)
    length = shape(wdata)

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a new file using the default properties.
    call h5fcreate_f(dataFileName, H5F_ACC_TRUNC_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 data file (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, length, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_integer, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_integer, wdata, length, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_info

  subroutine write_int_n(Array, length, datasetname)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    integer, intent(in) :: length
    integer, dimension(length), intent(in) :: Array
    character(len=*), intent(in) :: datasetname

    integer, parameter :: rank = 1
    integer(HSIZE_T), dimension(rank) :: lenArr
    integer :: hdferr
    integer(HID_T) :: file_id, space_id, dset ! Handles  

    lenArr = shape(Array)

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    call h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_integer, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_integer, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_int_n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_int_n

  subroutine write_double_2n(Array, length1, length2, datasetname)
    use HDF5
    use quick_files_module, only: dataFileName

    implicit none

    integer, intent(in) :: length1, length2
    double precision, dimension(length1, length2), intent(in) :: Array
    character(len=*), intent(in) :: datasetname

    integer, parameter :: rank = 2
    integer(HSIZE_T), dimension(rank) :: lenArr
    integer :: hdferr
    logical :: exists
    integer(HID_T) :: file_id, space_id, dset ! Handles  

    lenArr = shape(Array)

    ! Initialize FORTRAN interface.
    call h5open_f(hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    call h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_DOUBLE, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_DOUBLE, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_double_2n)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_double_2n

#endif

end module quick_restart_module
