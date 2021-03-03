# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# AMBER: adapted from CheckSymbolExists.  Amazingly, this functionality is not available in the CMake standard library.

#[=======================================================================[.rst:
CheckConstantExists
-----------------

Provides a macro to check if something exists as a predefined constant or enum value in ``C``.

.. command:: check_constant_exists

	::

		check_constant_exists(<constant> <files> <constant>)

	Check that the ``<constant>`` globally defined variable OR enum
	is available after including given header ``<files>`` and store the
	 result in a ``<variable>``.  Specify the list
	of files in one argument as a semicolon-separated list.
	``<variable>`` will be created as an internal cache variable.
	If the symbol is a global constant, then it must be available for linking.

The following variables may be set before calling this macro to modify
the way the check is run:

``CMAKE_REQUIRED_FLAGS``
	string of compile command line flags
``CMAKE_REQUIRED_DEFINITIONS``
	list of macros to define (-DFOO=bar)
``CMAKE_REQUIRED_INCLUDES``
	list of include directories
``CMAKE_REQUIRED_LIBRARIES``
	list of libraries to link
``CMAKE_REQUIRED_QUIET``
	execute quietly without messages
#]=======================================================================]

macro(CHECK_CONSTANT_EXISTS CONSTANT FILES VARIABLE)
	if(CMAKE_C_COMPILER_LOADED)
		_CHECK_CONSTANT_EXISTS("${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/CheckConstantExists.c" "${CONSTANT}" "${FILES}" "${VARIABLE}" )
	elseif(CMAKE_CXX_COMPILER_LOADED)
		_CHECK_CONSTANT_EXISTS("${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/CheckConstantExists.cxx" "${CONSTANT}" "${FILES}" "${VARIABLE}" )
	else()
		message(FATAL_ERROR "CHECK_CONSTANT_EXISTS needs either C or CXX language enabled")
	endif()
endmacro()

macro(_CHECK_CONSTANT_EXISTS SOURCEFILE CONSTANT FILES VARIABLE)
	if(NOT DEFINED "${VARIABLE}" OR "x${${VARIABLE}}" STREQUAL "x${VARIABLE}")
		set(CMAKE_CONFIGURABLE_FILE_CONTENT "/* */\n")
		set(MACRO_CHECK_CONSTANT_EXISTS_FLAGS ${CMAKE_REQUIRED_FLAGS})
		if(CMAKE_REQUIRED_LIBRARIES)
			set(CHECK_CONSTANT_EXISTS_LIBS
				LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
		else()
			set(CHECK_CONSTANT_EXISTS_LIBS)
		endif()
		if(CMAKE_REQUIRED_INCLUDES)
			set(CMAKE_CONSTANT_EXISTS_INCLUDES
				"-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}")
		else()
			set(CMAKE_CONSTANT_EXISTS_INCLUDES)
		endif()
		foreach(FILE ${FILES})
			set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT} #include <${FILE}>\n")
		endforeach()
		
		set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}
int main(int argc, char** argv)
{
	(void)argv;
	return argc = ${CONSTANT};
}
")

		configure_file("${CMAKE_ROOT}/Modules/CMakeConfigurableFile.in"
			"${SOURCEFILE}" @ONLY)

		if(NOT CMAKE_REQUIRED_QUIET)
			message(STATUS "Looking for ${CONSTANT} as #define or enum")
		endif()
		try_compile(${VARIABLE}
			${CMAKE_BINARY_DIR}
			"${SOURCEFILE}"
			COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
			${CHECK_CONSTANT_EXISTS_LIBS}
			CMAKE_FLAGS
			-DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_CONSTANT_EXISTS_FLAGS}
			"${CMAKE_CONSTANT_EXISTS_INCLUDES}"
			OUTPUT_VARIABLE OUTPUT)
		if(${VARIABLE})
			if(NOT CMAKE_REQUIRED_QUIET)
				message(STATUS "Looking for ${CONSTANT} as #define or enum - found")
			endif()
			set(${VARIABLE} 1 CACHE INTERNAL "Have constant ${CONSTANT}")
			file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
				"Determining if the ${CONSTANT} "
				"exist passed with the following output:\n"
				"${OUTPUT}\nFile ${SOURCEFILE}:\n"
				"${CMAKE_CONFIGURABLE_FILE_CONTENT}\n")
		else()
			if(NOT CMAKE_REQUIRED_QUIET)
				message(STATUS "Looking for ${CONSTANT} as #define or enum - not found")
			endif()
			set(${VARIABLE} "" CACHE INTERNAL "Have constant ${CONSTANT}")
			file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
				"Determining if the ${CONSTANT} "
				"exist failed with the following output:\n"
				"${OUTPUT}\nFile ${SOURCEFILE}:\n"
				"${CMAKE_CONFIGURABLE_FILE_CONTENT}\n")
		endif()
	endif()
endmacro()