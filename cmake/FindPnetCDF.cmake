# - Find PnetCDF, the Parallel NetCDF library
#
# Currently only the C interface is found and tested since that's all that Amber uses, though the 
# Fortran one could also be found with a bit more work.
#
# This module expects that you have already called FindMPI for the C language, and will test linking PnetCDF
# together with MPI.
#
# This module defines the following variables:
#
#  PnetCDF_LIBRARY, the library needed to use PnetCDF.  You will likely also want to link MPI to targets that use PnetCDF.
#  PnetCDF_VERSION, version of PnetCDF that was found
#  PnetCDF_FOUND, If false, do not try to use PnetCDF.
#  PnetCDF_INCLUDE_DIR, directory to include so that you can #include <pnetcdf.h>
#
# If PnetCDF was found, this module also creates the following imported target:
#  pnetcdf::pnetcdf - Target to use the PnetCDF library.  You will likely also want to link MPI to targets that use PnetCDF.


# find library
find_library(PnetCDF_LIBRARY
    NAMES pnetcdf
    DOC "Path to PnetCDF library")

# search for headers
find_path(PnetCDF_INCLUDE_DIR
    NAMES pnetcdf.h
    DOC "Path to PnetCDF's includes.  Should contain pnetcdf.h")

# detect version
if(PnetCDF_INCLUDE_DIR)
    
    # read first 1000 chars of main header
    file(READ ${PnetCDF_INCLUDE_DIR}/pnetcdf.h PNETCDF_H_CONTENT LIMIT 1000)

    if("${PNETCDF_H_CONTENT}" MATCHES "#define PNETCDF_VERSION +\"([0-9.]+)\"")
        set(PnetCDF_VERSION ${CMAKE_MATCH_1})
    endif()
endif()

# check functionality
if(MPI_C_FOUND)
    include(CMakePushCheckState)
    cmake_push_check_state()

    set(CMAKE_REQUIRED_FLAGS ${MPI_C_COMPILE_FLAGS})
    try_link_library(PnetCDF_WORKS
        LANGUAGE C
        FUNCTION ncmpi_strerror
        LIBRARIES ${PnetCDF_LIBRARY} ${MPI_C_LIBRARIES}
        INCLUDES ${PnetCDF_INCLUDE_DIR} ${MPI_C_INCLUDE_PATH}
        FUNC_DECLARATION "#include <pnetcdf.h>"
        FUNC_CALL "ncmpi_strerror(0)")

    cmake_pop_check_state()
endif()

find_package_handle_standard_args(PnetCDF
    REQUIRED_VARS PnetCDF_LIBRARY PnetCDF_INCLUDE_DIR PnetCDF_WORKS
    VERSION_VAR PnetCDF_VERSION)

# create imported target
if(PnetCDF_FOUND)
    add_library(pnetcdf::pnetcdf UNKNOWN IMPORTED)
    set_property(TARGET pnetcdf::pnetcdf PROPERTY IMPORTED_LOCATION ${PnetCDF_LIBRARY})
    set_property(TARGET pnetcdf::pnetcdf PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PnetCDF_INCLUDE_DIR})
endif()