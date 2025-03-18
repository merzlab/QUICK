# Checks environment for hdf5 library installation. 

find_package(HDF5 COMPONENTS Fortran HL)
if(HDF5_FOUND)
  include_directories(${HDF5_INCLUDE_DIRS})
#  list_to_space_separated(HDF5_LIBRARIES_SPC ${HDF5_LIBRARIES})
  message(STATUS "Found HDF5: ${HDF5_VERSION}")
else()
  message(FATAL_ERROR "HDF5 is requested but was not found. Please, make sure HDF5 is installed.")
endif()

