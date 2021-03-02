#File to search for miscellaneous programs that Amber uses
#NOTE: must be included after MPIConfig, OpenMPConfig, and CompilerFLags


#------------------------------------------------------------------------------
# check for dl
#------------------------------------------------------------------------------
check_library_exists(dl dlopen "" HAVE_LIBDL)

#-----------------------------------
# M4 for XLeap
#-----------------------------------
find_program(M4 m4)

#------------------------------------------------------------------------------
#  Flex
#------------------------------------------------------------------------------
find_package(FLEX REQUIRED)

#------------------------------------------------------------------------------
#  Bison
#------------------------------------------------------------------------------
find_package(BISON REQUIRED)

#------------------------------------------------------------------------------
#  bash, for running shell scripts
#------------------------------------------------------------------------------
find_program(BASH bash)

if(BASH AND NOT HOST_WINDOWS)
	set(AUTOMAKE_DEFAULT TRUE)
else()
	set(AUTOMAKE_DEFAULT FALSE)
endif()

option(CAN_BUILD_AUTOMAKE "Whether it is possible to build dependencies which use Automake build systems on this platform." ${AUTOMAKE_DEFAULT}) 

#------------------------------------------------------------------------------
#  make, for configure scripts
#------------------------------------------------------------------------------
set(MAKE_COMMAND make CACHE STRING "Command to run to make 3rd party projects with autotools / make-based build systems.")

