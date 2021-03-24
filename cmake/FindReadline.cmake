# from http://websvn.kde.org/trunk/KDE/kdeedu/cmake/modules/FindReadline.cmake
# http://websvn.kde.org/trunk/KDE/kdeedu/cmake/modules/COPYING-CMAKE-SCRIPTS
# --> BSD licensed
#
# Modified for AMBER
#
# GNU Readline library finder.
#
# Variables: 
#   READLINE_INCLUDE_DIR - directory containing readline/readline.h
#   READLINE_LIBRARY - library to link for readline
#   READLINE_COMPILE_DEFINITIONS - Compile definitions needed for dll imports on Windows
#
# If readline was found, this module also creates the following imported target:
#  readline::readline - Target for readline library

include(CMakePushCheckState)
cmake_push_check_state()

find_path(READLINE_INCLUDE_DIR NAMES readline/readline.h DOC "directory containing readline/readline.h")

find_library(READLINE_LIBRARY NAMES readline DOC "Path to readline library.")

if(EXISTS "${READLINE_LIBRARY}")

	# Configure dll imports if necessary
	set(READLINE_COMPILE_DEFINITIONS "")

	if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
		get_lib_type(${READLINE_LIBRARY} READLINE_LIB_TYPE)
		if("${READLINE_LIB_TYPE}" STREQUAL "STATIC")
			set(READLINE_COMPILE_DEFINITIONS USE_READLINE_STATIC)
		else()
			set(READLINE_COMPILE_DEFINITIONS USE_READLINE_DLL)
		endif()
	endif()

	set(CMAKE_REQUIRED_DEFINITIONS ${READLINE_COMPILE_DEFINITIONS})

	# now check if the library we found actually works (certain versions of Anaconda ship with a broken 
	# libreadline.so that uses functions from libtinfo, but does not declare a dynamic dependency on said library.)
	try_link_library(READLINE_WORKS
		LANGUAGE C
		FUNCTION rl_initialize
		LIBRARIES ${READLINE_LIBRARY}
		INCLUDES ${READLINE_INCLUDE_DIR}
		FUNC_DECLARATION "#include <readline/readline.h>")
endif()

mark_as_advanced(READLINE_INCLUDE_DIR READLINE_LIBRARY READLINE_WORKS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Readline DEFAULT_MSG READLINE_INCLUDE_DIR READLINE_LIBRARY READLINE_WORKS)

# create imported target
if(READLINE_FOUND)
	add_library(readline::readline UNKNOWN IMPORTED)
    set_property(TARGET readline::readline PROPERTY IMPORTED_LOCATION ${READLINE_LIBRARY})
    set_property(TARGET readline::readline PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${READLINE_INCLUDE_DIR})
    set_property(TARGET readline::readline PROPERTY INTERFACE_COMPILE_DEFINITIONS ${READLINE_COMPILE_DEFINITIONS})
endif()

cmake_pop_check_state()