# - Find nlopt
# 
# This module finds the NLOPT library on your system.
#
# Variables: 
#  nlopt_LIBRARIES - Libraries to link to use nlopt as a shared library
#  nlopt_INCLUDES - Path where nlopt's headers can be found
#  nlopt_FOUND - Whether nlopt was found
#  
# If nlopt was found, this module also creates the following imported target:
#  nlopt::nlopt - Target for nlopt's libraries


# if(DEFINED MINICONDA_INSTALL_DIR)
#   find_library(nlopt_LIBRARY
#     NAMES nlopt
#     PATHS ${MINICONDA_INSTALL_DIR}/lib
#     DOC "Path to nlopt library")
  
#   find_path(nlopt_INCLUDE_DIR
#     NAMES nlopt.h nlopt.hpp
#     PATHS ${MINICONDA_INSTALL_DIR}/include
#     DOC "Path to nlopt's includes.  Should contain nlopt.h and nlopt.hpp")
# else()

  find_library(nlopt_LIBRARY
    NAMES nlopt
    DOC "Path to nlopt library")
  
  find_path(nlopt_INCLUDE_DIR
    NAMES nlopt.h nlopt.hpp
    DOC "Path to nlopt's includes.  Should contain nlopt.h and nlopt.hpp")
  
# endif()



find_package_handle_standard_args(nlopt
    REQUIRED_VARS nlopt_LIBRARY nlopt_INCLUDE_DIR)

  
# create imported target
if(nlopt_FOUND)
    message("-- Found external nlopt library ${nlopt_LIBRARY}")
    message("-- Found external nlopt include directory ${nlopt_INCLUDE_DIR}")

    add_library(nlopt::nlopt UNKNOWN IMPORTED)
    set_property(TARGET nlopt::nlopt PROPERTY IMPORTED_LOCATION ${nlopt_LIBRARY})
    set_property(TARGET nlopt::nlopt PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${nlopt_INCLUDE_DIR})
endif()
