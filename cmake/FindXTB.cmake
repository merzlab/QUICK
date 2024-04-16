# - Find XTB
# 
# This module finds the XTB library on your system.
#
# Variables: 
#  XTB_LIBRARIES - Libraries to link to use XTB as a shared library
#  XTB_INCLUDES - Path where XTB's headers can be found
#  XTB_FOUND - Whether XTB was found
#  
# If XTB was found, this module also creates the following imported target:
#  XTB::XTB - Target for XTB's libraries


# find library
find_library(XTB_LIBRARY
    NAMES xtb
    DOC "Path to XTB library")

# search for headers
find_path(XTB_INCLUDE_DIR
    NAMES xtb.h
    DOC "Path to XTB's includes.  Should contain xtb.h and fortran modules; e.g., xtb_xtb_calculator.mod")

#find_path(XTB_BUILD_DIR
#	NAMES include/xtb_xtb_calculator.mod
#	DOC "Path to XTB build directory containing the uninstalled XTB fortran modules, e.g. include/xtb_xtb_calculator.mod")

find_package_handle_standard_args(XTB
    REQUIRED_VARS XTB_LIBRARY XTB_INCLUDE_DIR) # XTB_BUILD_DIR

# create imported target
if(XTB_FOUND)
    add_library(xtb::xtb UNKNOWN IMPORTED)
    set_property(TARGET xtb::xtb PROPERTY IMPORTED_LOCATION ${XTB_LIBRARY})
    set_property(TARGET xtb::xtb PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${XTB_INCLUDE_DIR})
    #set_property(TARGET xtb::xtb PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${XTB_BUILD_DIR}/include;${XTB_INCLUDE_DIR}")
endif()
