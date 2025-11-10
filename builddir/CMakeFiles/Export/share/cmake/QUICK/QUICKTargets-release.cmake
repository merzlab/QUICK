#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "QUICK::blas" for configuration "RELEASE"
set_property(TARGET QUICK::blas APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::blas PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libblas-quick.so"
  IMPORTED_SONAME_RELEASE "libblas-quick.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::blas )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::blas "${_IMPORT_PREFIX}/lib/libblas-quick.so" )

# Import target "QUICK::lapack" for configuration "RELEASE"
set_property(TARGET QUICK::lapack APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::lapack PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblapack-quick.so"
  IMPORTED_SONAME_RELEASE "liblapack-quick.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::lapack )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::lapack "${_IMPORT_PREFIX}/lib/liblapack-quick.so" )

# Import target "QUICK::libquick" for configuration "RELEASE"
set_property(TARGET QUICK::libquick APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::libquick PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "QUICK::lapack;QUICK::blas"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libquick.so"
  IMPORTED_SONAME_RELEASE "libquick.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::libquick )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::libquick "${_IMPORT_PREFIX}/lib/libquick.so" )

# Import target "QUICK::libquick_mpi" for configuration "RELEASE"
set_property(TARGET QUICK::libquick_mpi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::libquick_mpi PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "QUICK::lapack;QUICK::blas"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libquick_mpi.so"
  IMPORTED_SONAME_RELEASE "libquick_mpi.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::libquick_mpi )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::libquick_mpi "${_IMPORT_PREFIX}/lib/libquick_mpi.so" )

# Import target "QUICK::libquick_cuda" for configuration "RELEASE"
set_property(TARGET QUICK::libquick_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::libquick_cuda PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "QUICK::lapack;QUICK::blas;cublas;cusolver"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libquick_cuda.so"
  IMPORTED_SONAME_RELEASE "libquick_cuda.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::libquick_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::libquick_cuda "${_IMPORT_PREFIX}/lib/libquick_cuda.so" )

# Import target "QUICK::libquick_mpi_cuda" for configuration "RELEASE"
set_property(TARGET QUICK::libquick_mpi_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::libquick_mpi_cuda PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "QUICK::lapack;QUICK::blas;cublas;cusolver"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libquick_mpi_cuda.so"
  IMPORTED_SONAME_RELEASE "libquick_mpi_cuda.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::libquick_mpi_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::libquick_mpi_cuda "${_IMPORT_PREFIX}/lib/libquick_mpi_cuda.so" )

# Import target "QUICK::quick" for configuration "RELEASE"
set_property(TARGET QUICK::quick APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::quick PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/quick"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::quick )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::quick "${_IMPORT_PREFIX}/bin/quick" )

# Import target "QUICK::test-api" for configuration "RELEASE"
set_property(TARGET QUICK::test-api APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::test-api PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/test-api"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::test-api )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::test-api "${_IMPORT_PREFIX}/bin/test-api" )

# Import target "QUICK::quick.MPI" for configuration "RELEASE"
set_property(TARGET QUICK::quick.MPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::quick.MPI PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/quick.MPI"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::quick.MPI )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::quick.MPI "${_IMPORT_PREFIX}/bin/quick.MPI" )

# Import target "QUICK::test-api.MPI" for configuration "RELEASE"
set_property(TARGET QUICK::test-api.MPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::test-api.MPI PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/test-api.MPI"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::test-api.MPI )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::test-api.MPI "${_IMPORT_PREFIX}/bin/test-api.MPI" )

# Import target "QUICK::quick.cuda" for configuration "RELEASE"
set_property(TARGET QUICK::quick.cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::quick.cuda PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/quick.cuda"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::quick.cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::quick.cuda "${_IMPORT_PREFIX}/bin/quick.cuda" )

# Import target "QUICK::test-api.cuda" for configuration "RELEASE"
set_property(TARGET QUICK::test-api.cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::test-api.cuda PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/test-api.cuda"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::test-api.cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::test-api.cuda "${_IMPORT_PREFIX}/bin/test-api.cuda" )

# Import target "QUICK::quick.cuda.MPI" for configuration "RELEASE"
set_property(TARGET QUICK::quick.cuda.MPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::quick.cuda.MPI PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/quick.cuda.MPI"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::quick.cuda.MPI )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::quick.cuda.MPI "${_IMPORT_PREFIX}/bin/quick.cuda.MPI" )

# Import target "QUICK::test-api.cuda.MPI" for configuration "RELEASE"
set_property(TARGET QUICK::test-api.cuda.MPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUICK::test-api.cuda.MPI PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/test-api.cuda.MPI"
  )

list(APPEND _IMPORT_CHECK_TARGETS QUICK::test-api.cuda.MPI )
list(APPEND _IMPORT_CHECK_FILES_FOR_QUICK::test-api.cuda.MPI "${_IMPORT_PREFIX}/bin/test-api.cuda.MPI" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
