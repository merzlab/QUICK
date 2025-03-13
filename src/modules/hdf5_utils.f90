! ---------------------------------------------------------------------------- !
!
! MODULE: hdf5_utils
!
! DESCRIPTION: Utilities for reading and writing HDF5 files in Fortran.
!
! AUTHORS: Daniel Mentiplay, David Liptai, Conrad Chan
!
! YEAR: 2019
!
! DEPENDENCIES: HDF5, ISO_C_BINDING
!
! ---------------------------------------------------------------------------- !
MODULE hdf5_utils

  USE HDF5
  USE ISO_C_BINDING, ONLY:C_LOC

  IMPLICIT NONE

  PRIVATE

  INTEGER, PARAMETER :: compression_level = 9

  PUBLIC :: write_to_hdf5,    &
            read_from_hdf5,   &
            open_hdf5file,    &
            create_hdf5file,  &
            close_hdf5file,   &
            open_hdf5group,   &
            create_hdf5group, &
            close_hdf5group,  &
            HID_T

  INTERFACE write_to_hdf5
    MODULE PROCEDURE write_real_kind4,              & ! REAL(4)
                     write_real_kind8,              & ! REAL(8)
                     write_real_1d_array_kind4,     & ! 1d REAL(4) arrays
                     write_real_1d_array_kind8,     & ! 1d REAL(8) arrays
                     write_real_2d_array_kind4,     & ! 2d REAL(4) arrays
                     write_real_2d_array_kind8,     & ! 2d REAL(8) arrays
                     write_real_3d_array_kind4,     & ! 3d REAL(4) arrays
                     write_real_3d_array_kind8,     & ! 3d REAL(8) arrays
                     write_integer_kind4,           & ! INTEGER(4)
                     write_integer_1d_array_kind1,  & ! 1d INTEGER(1) arrays
                     write_integer_1d_array_kind4,  & ! 1d INTEGER(4) arrays
                     write_string                     ! strings
  END INTERFACE write_to_hdf5

  INTERFACE read_from_hdf5
    MODULE PROCEDURE read_real_kind4,              & ! REAL(4)
                     read_real_kind8,              & ! REAL(8)
                     read_real_1d_array_kind4,     & ! 1d REAL(4) arrays
                     read_real_1d_array_kind8,     & ! 1d REAL(8) arrays
                     read_real_2d_array_kind4,     & ! 2d REAL(4) arrays
                     read_real_2d_array_kind8,     & ! 2d REAL(8) arrays
                     read_real_3d_array_kind4,     & ! 3d REAL(4) arrays
                     read_real_3d_array_kind8,     & ! 3d REAL(8) arrays
                     read_integer_kind4,           & ! INTEGER(4)
                     read_integer_1d_array_kind1,  & ! 1d INTEGER(1) arrays
                     read_integer_1d_array_kind4,  & ! 1d INTEGER(4) arrays
                     read_string                     ! strings
  END INTERFACE read_from_hdf5

CONTAINS

SUBROUTINE create_hdf5group(file_id, groupname, group_id, error)
  CHARACTER(LEN=*), INTENT(IN)  :: groupname
  INTEGER(HID_T),   INTENT(IN)  :: file_id
  INTEGER(HID_T),   INTENT(OUT) :: group_id
  INTEGER,          INTENT(OUT) :: error

  CALL H5GCREATE_F(file_id, groupname, group_id, error)

END SUBROUTINE create_hdf5group

SUBROUTINE open_hdf5group(file_id, groupname, group_id, error)
  CHARACTER(LEN=*), INTENT(IN)  :: groupname
  INTEGER(HID_T),   INTENT(IN)  :: file_id
  INTEGER(HID_T),   INTENT(OUT) :: group_id
  INTEGER,          INTENT(OUT) :: error

  CALL H5GOPEN_F(file_id, groupname, group_id, error)

END SUBROUTINE open_hdf5group

SUBROUTINE close_hdf5group(group_id, error)
  INTEGER(HID_T),   INTENT(IN)  :: group_id
  INTEGER,          INTENT(OUT) :: error

  CALL H5GCLOSE_F(group_id, error)

END SUBROUTINE close_hdf5group

SUBROUTINE create_hdf5file(filename, file_id, error)
  CHARACTER(LEN=*), INTENT(IN)  :: filename
  INTEGER(HID_T),   INTENT(OUT) :: file_id
  INTEGER,          INTENT(OUT) :: error
  INTEGER :: filter_info
  INTEGER :: filter_info_both
  LOGICAL :: avail

  ! Initialise HDF5
  CALL H5OPEN_F(error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot initialise HDF5",/)')
    RETURN
  ENDIF

  ! Check if gzip compression is available.
  CALL H5ZFILTER_AVAIL_F(H5Z_FILTER_DEFLATE_F, avail, error)
  IF (.NOT.avail) THEN
    WRITE(*,'("gzip filter not available.",/)')
    RETURN
  ENDIF
  CALL H5ZGET_FILTER_INFO_F(H5Z_FILTER_DEFLATE_F, filter_info, error)
  filter_info_both=IOR(H5Z_FILTER_ENCODE_ENABLED_F, H5Z_FILTER_DECODE_ENABLED_F)
  IF (filter_info /= filter_info_both) THEN
    WRITE(*,'("gzip filter not available for encoding and decoding.",/)')
    RETURN
  ENDIF

  ! Create file
  CALL H5FCREATE_F(filename, H5F_ACC_TRUNC_F, file_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 file",/)')
    RETURN
  ENDIF

END SUBROUTINE create_hdf5file

SUBROUTINE open_hdf5file(filename, file_id, error)
  CHARACTER(LEN=*), INTENT(IN)  :: filename
  INTEGER(HID_T),   INTENT(OUT) :: file_id
  INTEGER,          INTENT(OUT) :: error

  ! Initialise HDF5
  CALL H5OPEN_F(error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot initialise HDF5",/)')
    RETURN
  ENDIF

  ! Open file
  CALL H5FOPEN_F(filename, H5F_ACC_RDWR_F, file_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 file",/)')
    RETURN
  ENDIF

END SUBROUTINE open_hdf5file

SUBROUTINE close_hdf5file(file_id, error)
  INTEGER(HID_T), INTENT(IN)  :: file_id
  INTEGER,        INTENT(OUT) :: error

  ! Close file
  CALL H5FCLOSE_F(file_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 file",/)')
    RETURN
  ENDIF

  ! Close HDF5
  CALL H5CLOSE_F(error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5",/)')
    RETURN
  ENDIF

END SUBROUTINE close_hdf5file

SUBROUTINE write_real_kind4(x, name, id, error)
  REAL(KIND=4),   INTENT(IN)  :: x
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dspace_id
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_REAL

  ! Create dataspace
  CALL H5SCREATE_F(H5S_SCALAR_F, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_kind4

SUBROUTINE write_real_kind8(x, name, id, error)
  REAL(KIND=8),   INTENT(IN)  :: x
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dspace_id
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_DOUBLE

  ! Create dataspace
  CALL H5SCREATE_F(H5S_SCALAR_F, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_kind8

SUBROUTINE write_real_1d_array_kind4(x, name, id, error)
  REAL(KIND=4),   INTENT(IN)  :: x(:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_1d_array_kind4

SUBROUTINE write_real_1d_array_kind8(x, name, id, error)
  REAL(KIND=8),   INTENT(IN)  :: x(:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_1d_array_kind8

SUBROUTINE write_real_2d_array_kind4(x, name, id, error)
  REAL(KIND=4),   INTENT(IN)  :: x(:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 2
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_2d_array_kind4

SUBROUTINE write_real_2d_array_kind8(x, name, id, error)
  REAL(KIND=8),   INTENT(IN)  :: x(:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 2
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_2d_array_kind8

SUBROUTINE write_real_3d_array_kind4(x, name, id, error)
  REAL(KIND=4),   INTENT(IN)  :: x(:,:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 3
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_3d_array_kind4

SUBROUTINE write_real_3d_array_kind8(x, name, id, error)
  REAL(KIND=8),   INTENT(IN)  :: x(:,:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 3
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_real_3d_array_kind8

SUBROUTINE write_integer_kind4(x, name, id, error)
  INTEGER(KIND=4), INTENT(IN)  :: x
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  INTEGER,         INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dspace_id
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_INTEGER

  ! Create dataspace
  CALL H5SCREATE_F(H5S_SCALAR_F, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_integer_kind4

SUBROUTINE write_integer_1d_array_kind4(x, name, id, error)
  INTEGER(KIND=4), INTENT(IN)  :: x(:)
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  INTEGER,         INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_NATIVE_INTEGER

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_integer_1d_array_kind4

SUBROUTINE write_integer_1d_array_kind1(x, name, id, error)
  INTEGER(KIND=1), INTENT(IN)  :: x(:)
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  INTEGER,         INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HSIZE_T)   :: chunk(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: prop_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  chunk = shape(x)
  dtype_id = H5T_STD_I8LE

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, xshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset creation property list, add the gzip
  ! compression filter and set the chunk size.
  CALL H5PCREATE_F(H5P_DATASET_CREATE_F, prop_id, error)
  CALL H5PSET_DEFLATE_F(prop_id, compression_level, error)
  CALL H5PSET_CHUNK_F(prop_id, ndims, chunk, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 property list",/)')
    RETURN
  ENDIF

  ! Create dataset in file
  CALL H5DCREATE_F(id, name, dtype_id, dspace_id, dset_id, error, prop_id)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Write to file
  CALL H5DWRITE_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close property list
  CALL H5PCLOSE_F(prop_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 property list",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

END SUBROUTINE write_integer_1d_array_kind1

SUBROUTINE write_string(str, name, id, error)
  CHARACTER(*),    INTENT(IN), TARGET :: str
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  INTEGER,         INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 0
  INTEGER(HSIZE_T)   :: sshape(ndims)
  INTEGER(HID_T)     :: dspace_id
  INTEGER(HID_T)     :: dset_id
  INTEGER(SIZE_T)    :: slength
  INTEGER(HID_T)     :: filetype
  TYPE(C_PTR)        :: cpointer

  slength = LEN(str)
  sshape  = shape(str)

  ! Create file datatypes. Save the string as FORTRAN string
  CALL H5TCOPY_F(H5T_FORTRAN_S1,filetype, error)
  CALL H5TSET_SIZE_F(filetype, slength, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 datatype",/)')
    RETURN
  ENDIF

  ! Create dataspace
  CALL H5SCREATE_SIMPLE_F(ndims, sshape, dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Create the dataset in file
  CALL H5DCREATE_F(id, name, filetype, dspace_id, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot create HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Find C pointer
  cpointer = C_LOC(str(1:1))

  ! Write to file
  CALL H5DWRITE_F(dset_id, filetype, cpointer, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot write to HDF5 file",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataspace
  CALL H5SCLOSE_F(dspace_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataspace",/)')
    RETURN
  ENDIF

  ! Close datatype
  CALL H5TCLOSE_F(filetype, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 datatype",/)')
    RETURN
  ENDIF

END SUBROUTINE write_string

SUBROUTINE read_real_kind4(x, name, id, got, error)
  REAL(KIND=4),   INTENT(OUT) :: x
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_REAL

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_kind4

SUBROUTINE read_real_kind8(x, name, id, got, error)
  REAL(KIND=8),   INTENT(OUT) :: x
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_DOUBLE

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_kind8

SUBROUTINE read_real_1d_array_kind4(x, name, id, got, error)
  REAL(KIND=4),   INTENT(OUT) :: x(:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_1d_array_kind4

SUBROUTINE read_real_1d_array_kind8(x, name, id, got, error)
  REAL(KIND=8),   INTENT(OUT) :: x(:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_1d_array_kind8

SUBROUTINE read_real_2d_array_kind4(x, name, id, got, error)
  REAL(KIND=4),   INTENT(OUT) :: x(:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 2
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_2d_array_kind4

SUBROUTINE read_real_2d_array_kind8(x, name, id, got, error)
  REAL(KIND=8),   INTENT(OUT) :: x(:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 2
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_2d_array_kind8

SUBROUTINE read_real_3d_array_kind4(x, name, id, got, error)
  REAL(KIND=4),   INTENT(OUT) :: x(:,:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 3
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_REAL

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_3d_array_kind4

SUBROUTINE read_real_3d_array_kind8(x, name, id, got, error)
  REAL(KIND=8),   INTENT(OUT) :: x(:,:,:)
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 3
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_DOUBLE

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_real_3d_array_kind8

SUBROUTINE read_integer_kind4(x, name, id, got, error)
  INTEGER(KIND=4), INTENT(OUT) :: x
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  LOGICAL,         INTENT(OUT) :: got
  INTEGER,         INTENT(OUT) :: error

  INTEGER(HSIZE_T), PARAMETER  :: xshape(0) = 0
  INTEGER(HID_T) :: dset_id
  INTEGER(HID_T) :: dtype_id

  dtype_id = H5T_NATIVE_INTEGER

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_integer_kind4

SUBROUTINE read_integer_1d_array_kind1(x, name, id, got, error)
  INTEGER(KIND=1), INTENT(OUT) :: x(:)
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  LOGICAL,         INTENT(OUT) :: got
  INTEGER,         INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_STD_I8LE

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_integer_1d_array_kind1

SUBROUTINE read_integer_1d_array_kind4(x, name, id, got, error)
  INTEGER(KIND=4), INTENT(OUT) :: x(:)
  CHARACTER(*),    INTENT(IN)  :: name
  INTEGER(HID_T),  INTENT(IN)  :: id
  LOGICAL,         INTENT(OUT) :: got
  INTEGER,         INTENT(OUT) :: error

  INTEGER, PARAMETER :: ndims = 1
  INTEGER(HSIZE_T)   :: xshape(ndims)
  INTEGER(HID_T)     :: dset_id
  INTEGER(HID_T)     :: dtype_id

  xshape = shape(x)
  dtype_id = H5T_NATIVE_INTEGER

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  ! Open dataset
  CALL H5DOPEN_F(id, name, dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Read dataset
  CALL H5DREAD_F(dset_id, dtype_id, x, xshape, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close dataset
  CALL H5DCLOSE_F(dset_id, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_integer_1d_array_kind4

SUBROUTINE read_string(str, name, id, got, error)
  CHARACTER(*),   INTENT(OUT) :: str
  CHARACTER(*),   INTENT(IN)  :: name
  INTEGER(HID_T), INTENT(IN)  :: id
  LOGICAL,        INTENT(OUT) :: got
  INTEGER,        INTENT(OUT) :: error

  INTEGER,         PARAMETER :: dim0 = 1
  INTEGER(SIZE_T), PARAMETER :: sdim = 100

  INTEGER(HSIZE_T) :: dims(1) = (/dim0/)
  INTEGER(HSIZE_T) :: maxdims(1)

  INTEGER(HID_T) :: filetype, memtype, space, dset

  CHARACTER(LEN=sdim), ALLOCATABLE, TARGET :: rdata(:)
  INTEGER(SIZE_T) :: size
  TYPE(C_PTR) :: f_ptr

  ! Check if dataset exists
  CALL H5LEXISTS_F(id, name, got, error)
  IF (.NOT.got) RETURN

  CALL H5DOPEN_F(id, name, dset, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot open HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Get the datatype and its size.
  CALL H5DGET_TYPE_F(dset, filetype, error)
  CALL H5TGET_SIZE_F(filetype, size, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot get HDF5 datatype or size",/)')
    RETURN
  ENDIF

  ! Make sure the declared length is large enough,
  ! the C string contains the null character.
  IF (size > sdim+1) THEN
    PRINT*,'ERROR: Character LEN is too small'
    STOP
  ENDIF

  ! Get dataspace.
  CALL H5DGET_SPACE_F(dset, space, error)
  CALL H5SGET_SIMPLE_EXTENT_DIMS_F(space, dims, maxdims, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot get HDF5 dataspace",/)')
    RETURN
  ENDIF

  ALLOCATE(rdata(1:dims(1)))

  ! Create the memory datatype.
  CALL H5TCOPY_F(H5T_FORTRAN_S1,memtype, error)
  CALL H5TSET_SIZE_F(memtype, sdim, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot get HDF5 memory datatype",/)')
    RETURN
  ENDIF

  ! Read the data.
  f_ptr = C_LOC(rdata(1)(1:1))
  CALL H5DREAD_F(dset, memtype, f_ptr, error,space)
  IF (error /= 0) THEN
    WRITE(*,'("cannot read HDF5 dataset",/)')
    RETURN
  ENDIF

  ! Close and release resources.
  CALL H5DCLOSE_F(dset, error)
  CALL H5SCLOSE_F(space, error)
  CALL H5TCLOSE_F(filetype, error)
  CALL H5TCLOSE_F(memtype, error)
  IF (error /= 0) THEN
    WRITE(*,'("cannot close HDF5 dataset",/)')
    RETURN
  ENDIF

  str = rdata(1)

  DEALLOCATE(rdata)

  IF (error /= 0) got = .FALSE.

END SUBROUTINE read_string

END MODULE hdf5_utils
