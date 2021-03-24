# AMBER: Originally based on Caffe's find module here: https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindNCCL.cmake

# FindNCCL.cmake: Finds the NVIDIA Collective Communications Library
#
# This module sets the following variables:
# NCCL_INCLUDE_DIR   - Path to include for NCCL
# NCCL_LIBRARY       - Library to link for NCCL
# NCCL_FOUND         - Whether NCCL was found
#
# If NCCL is found, it also creates the "nccl" imported target which is set
# up to use these paths.

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS $ENV{NCCL_HOME}/include)
find_library(NCCL_LIBRARY NAMES nccl nccl-static PATHS $ENV{NCCL_HOME}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
    add_library(nccl UNKNOWN IMPORTED)
    set_property(TARGET nccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR})
    set_property(TARGET nccl PROPERTY IMPORTED_LOCATION ${NCCL_LIBRARY})
endif()