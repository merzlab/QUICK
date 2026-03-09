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
! This module contains I/O utility routines for several files types,  !
! including checkpoint and data (DAT) file types.                     !
!---------------------------------------------------------------------!

#include "util.fh"

module quick_io_module

  implicit none

  public :: write_int_rank0, read_int_rank0
  public :: write_real8_rank0, read_real8_rank0  
  public :: write_int_rank3, read_int_rank3
  public :: write_real8_rank3, read_real8_rank3

#if defined(RESTART_HDF5)
  public :: read_hdf5_real8_rank2
  public :: read_hdf5_int_rank0
  public :: read_hdf5_int_rank1
  public :: write_hdf5_info
  public :: write_hdf5_int_rank1
  public :: write_hdf5_real8_rank2
#endif

contains

#if defined(RESTART_HDF5)
  subroutine read_hdf5_real8_rank2(datasetname, ind, n, adata)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a simple datapsace
    call h5screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(abuf(n(1),n(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, abuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    adata = abuf

    if(allocated(abuf)) deallocate(abuf)

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_hdf5_real8_rank2

  subroutine read_hdf5_int_rank0(datasetname, ind, idata)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_elements_f(file_space, H5S_SELECT_SET_F, rank, npoints, dims, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a scalar datapsace
    call h5screate_f(H5S_SCALAR_F, space_id, hdferr)

    ! Read the array in dataset "datasetname"
    call h5dread_f(dset, H5T_NATIVE_integer, ibuf, dimn, hdferr, space_id, file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata = ibuf(1,1)

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_hdf5_int_rank0)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_hdf5_int_rank0

  subroutine read_hdf5_int_rank1(datasetname, ind, n, idata)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file
    call h5fopen_f(dataFileName, H5F_ACC_RDONLY_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! open dataset
    call h5dopen_f(file_id, datasetname, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 dataset (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Get the file_space
    call h5dget_space_f(dset, file_space, hdferr)
    call h5sselect_hyperslab_f(file_space, 0, start, stride, hdferr, countn, blockn)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error getting space in the HDF5 dataset (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! create a simple datapsace
    call h5screate_simple_f(rank, dims, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Read the array in dataset "datasetname"
    allocate(ibuf(n))
    call h5dread_f(dset, H5T_NATIVE_integer, ibuf, dims, hdferr, mem_space_id=space_id, file_space_id=file_space)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error reading data from dataset (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! The data in the buffer is transferred to return to the calling program
    idata = ibuf

    if(allocated(ibuf)) deallocate(ibuf)

    ! Close file and dataset
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (read_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine read_hdf5_int_rank1

  subroutine write_hdf5_info(natom, nbasis)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a new file using the default properties.
    call h5fcreate_f(dataFileName, H5F_ACC_TRUNC_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 data file (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, length, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_integer, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_integer, wdata, length, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_hdf5_info)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_hdf5_info

  subroutine write_hdf5_int_rank1(Array, length, datasetname)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    call h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_integer, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_integer, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_hdf5_int_rank1)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_hdf5_int_rank1

  subroutine write_hdf5_real8_rank2(Array, length1, length2, datasetname)
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
      call PrtErr(OUTFILEHANDLE, 'Error initializing HDF5 Fortran interface (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Open file.
    call h5fopen_f(dataFileName, H5F_ACC_RDWR_F, file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to open HDF5 data file (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create a simple dataspace
    call h5screate_simple_f(rank, lenArr, space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error creating space in the HDF5 dataset (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Create dataset
    call h5dcreate_f(file_id, datasetname, H5T_NATIVE_DOUBLE, space_id, dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Failed to create HDF5 dataset (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Write the array to a dataset "datasetname"
    call h5dwrite_f(dset, H5T_NATIVE_DOUBLE, Array, lenArr, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error writing data to dataset (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataset
    call h5dclose_f(dset, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataset (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close dataspace
    call h5sclose_f(space_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 dataspace (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

    ! Close file and interface
    call h5fclose_f(file_id, hdferr)
    if (hdferr /= 0) then
      call PrtErr(OUTFILEHANDLE, 'Error closing HDF5 interface (write_hdf5_real8_rank2)')
      call quick_exit(OUTFILEHANDLE, 1)
    endif

  end subroutine write_hdf5_real8_rank2

#endif

  ! write one int value to chk file
  subroutine write_int_rank0(chk,key,nvalu,fail)
     implicit none
     integer chk,nvalu,i,j,k,l,fail
     character kline*40,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     write(chk) '#'//kline(1:40)
     write(chk) 'I '
     write(chk) nvalu
  
  end
  
  
  ! read one int value from chk file
  subroutine read_int_rank0(chk,key,nvalu,fail)
     implicit none
     integer chk,nvalu,i,j,k,l,num,fail
     character kline*40,ktype*2,line*41,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     fail=0
     rewind(chk)
     do
        read(chk,end=100,err=120) line
        if (line(1:1).ne.'#') cycle
        if (index(line,kline)==0) cycle
        read(chk,end=100,err=100) ktype
        if (ktype(1:1).ne.'I') exit
        read(chk,end=100,err=100) nvalu
        fail=1
        exit
     120    continue
     enddo
  
     100  return
  
  end
  
  
  ! write one double value to chk file
  subroutine write_real8_rank0(chk,key,dvalu,fail)
     implicit none
     integer chk,i,j,k,l,fail
     character kline*40,key*(*)
     real*8 dvalu
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     fail=0
     do
        read(chk,end=100,err=200)
     enddo
  
     100  write(chk) '#'//kline(1:40)
     write(chk) 'R '
     write(chk) dvalu
     fail=1
     200  return
  
  end
  
  
  ! read one double value from chk file
  subroutine read_real8_rank0(chk,key,dvalu,fail)
     implicit none
     integer chk,nvalu,i,j,k,l,num,fail
     character kline*40,ktype*2,line*41,key*(*)
     real*8 dvalu
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     fail=0
     rewind(chk)
     do
        read(chk,end=100,err=120) line
        if (line(1:1).ne.'#') cycle
        if (index(line,kline)==0) cycle
        read(chk,end=100,err=100) ktype
        if (ktype(1:1).ne.'R') exit
        read(chk,end=100,err=100) dvalu
        fail=1
        exit
     120    continue
     enddo
  
     100  return
  
  end
  
  
  ! write one int array to chk file
  subroutine write_int_rank3(chk,key,x,y,z,dim,fail)
     implicit none
     integer chk,x,y,z,i,j,k,l,fail
     integer dim(x,y,z)
     character kline*40,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     write(chk) '#'//kline(1:40)
     write(chk) 'II'
     write(chk) x*y*z
     write(chk) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
     fail=1
     200  return
  
  end
  
  
  ! read one int array from chk file
  subroutine read_int_rank3(chk,key,x,y,z,dim,fail)
     implicit none
     integer chk,x,y,z,i,j,k,l,fail
     integer num
     integer dim(x,y,z),dim_t(2*x,2*y,2*z)
     character kline*40,ktype*2,line*41,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     fail=0
     rewind(chk)
     do
        read(chk,end=100,err=120) line
        if (line(1:1).ne.'#') cycle
        if (index(line,kline)==0) cycle
        read(chk,end=100,err=100) ktype
        if (ktype.ne.'II') exit
        read(chk,end=100,err=100) num
        if (num.ne.x*y*z) exit
        read(chk,end=100,err=100) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
  
        fail=1
        exit
     120    continue
     enddo
  
     100  return
  
  end
  
  ! write one double array to chk file
  subroutine write_real8_rank3(chk,key,x,y,z,dim,fail)
     implicit none
     integer chk,x,y,z,i,j,k,l,fail
     real*8 dim(x,y,z)
     character kline*40,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     write(chk) '#'//kline(1:40)
     write(chk) 'RR'
     write(chk) x*y*z
     write(chk) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
     return
  
  end
  
  
  ! read one double array from chk file
  subroutine read_real8_rank3(chk,key,x,y,z,dim,fail)
     implicit none
     integer chk,x,y,z,i,j,k,l,num,fail
     real*8 dim(x,y,z)
     character kline*40,ktype*2,line*41,key*(*)
  
     l=len(key)
     if (l>=40) then
        kline=key(1:40)
     else
        kline(1:l)=key(1:l)
        do k=l+1,40
           kline(k:k)=' '
        enddo
     endif
  
     fail=0
     rewind(chk)
     do
        read(chk,end=100,err=120) line
        if (line(1:1).ne.'#') cycle
        if (index(line,kline)==0) cycle
        read(chk,end=100,err=100) ktype
        if (ktype.ne.'RR') exit
        read(chk,end=100,err=100) num
        if (num.ne.x*y*z) exit
        read(chk,end=100,err=100) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
        fail=1
        exit
     120  continue
     enddo
  
     100  return
  
  end

end module quick_io_module
