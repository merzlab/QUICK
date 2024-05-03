# - Find DeePMD-kit
# 
# This module finds the DeePMD-kit C library on your system.
#
# Variables:
#  DeePMD_FOUND - Whether DeePMD-kit was found
#  
# If DeePMD-kit was found, this module also creates the following imported target:
#  DEEPMD::deepmd_c - Target for DeePMD-kit's C libraries


# find library
find_library(DeePMD_LIBRARY
    NAMES deepmd_c
    DOC "Path to DeePMD-kit C library")

# search for headers
find_path(DeePMD_INCLUDE_DIR
    NAMES deepmd/deepmd.hpp deepmd/c_api.h
    DOC "Path to DeePMD-kit's includes.  Should contain deepmd/deepmd.hpp and deepmd/c_api.h")

find_package_handle_standard_args(DeePMD
    REQUIRED_VARS DeePMD_LIBRARY DeePMD_INCLUDE_DIR)

# create imported target
if(DeePMD_FOUND)
    add_library(DeePMD::deepmd_c UNKNOWN IMPORTED)
    set_property(TARGET DeePMD::deepmd_c PROPERTY IMPORTED_LOCATION ${DeePMD_LIBRARY})
    set_property(TARGET DeePMD::deepmd_c PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DeePMD_INCLUDE_DIR})
endif()
