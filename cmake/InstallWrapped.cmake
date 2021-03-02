# Script to handle installing certain programs with a wrapper script that sets AMBERHOME.

option(INSTALL_WRAPPED "If true, replace certain programs with a wrapper script that sets AMBERHOME automatically before calling the program.  Not supported on Windows with CMake older than 3.15." TRUE)

# Installs the given excutable targets via a wrapper script that calls amber.sh.
#usage: install_executables_wrapped(TARGETS foo bar [COMPONENT MPI])
# If COMPONENT is not given the default component is used.
function(install_executables_wrapped)
	
	# parse arguments
	cmake_parse_arguments(IEW
		""
		"COMPONENT"
		"TARGETS"
		${ARGN})

	if("${IEW_COMPONENT}" STREQUAL "")
		set(IEW_COMPONENT ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})
	endif()

	if("${IEW_COMPONENT}" STREQUAL "TARGETS")
		message(FATAL_ERROR "Incorrect usage.  No TARGETS given.")
	endif()

	# figure out if we need to actually do a wrapped install
	set(SKIP_WRAPPED_INSTALL FALSE)

	if(NOT INSTALL_WRAPPED)
		set(SKIP_WRAPPED_INSTALL TRUE)
	endif()

	if(TARGET_WINDOWS AND ${CMAKE_VERSION} VERSION_LESS 3.15)

		# no way to do a wrapped install without the TARGET_FILE_BASE_NAME
		# generator expression.
		set(SKIP_WRAPPED_INSTALL TRUE)
	endif()

	if(SKIP_WRAPPED_INSTALL)
		# simply do a regular install command
		install(TARGETS ${IEW_TARGETS} DESTINATION ${BINDIR})
		return()
	endif()

	set(SCRIPT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/wrapper_scripts)
	file(MAKE_DIRECTORY ${SCRIPT_DIR})

	foreach(EXECUTABLE ${IEW_TARGETS})

		# create script
		if(HOST_WINDOWS)
			# note: must use TARGET_FILE_BASE_NAME because we need to remove the .exe
			# and replace it with .bat for the program to be invoked correctly.
			set(WRAPPER_SCRIPT ${SCRIPT_DIR}/$<TARGET_FILE_BASE_NAME:${EXECUTABLE}>.bat)

			file(GENERATE OUTPUT ${WRAPPER_SCRIPT} CONTENT
"@echo off

rem AMBER wrapper script for the program ${EXECUTABLE}.
rem Calls the real ${EXECUTABLE} after setting needed environment variables.

set this_script_dir=%~dp0
set this_script_dir=%this_script_dir:~0,-1%
call %this_script_dir%\\..\\amber.bat

%AMBERHOME%\\bin\\wrapped_progs\\$<TARGET_FILE_NAME:${EXECUTABLE}> %*")


		else()
			set(WRAPPER_SCRIPT ${SCRIPT_DIR}/$<TARGET_FILE_NAME:${EXECUTABLE}>)

			file(GENERATE OUTPUT ${WRAPPER_SCRIPT} CONTENT
"#!/bin/bash

# AMBER wrapper script for the program ${EXECUTABLE}.
# Calls the real ${EXECUTABLE} after setting needed environment variables.

this_script_dir=\"$(cd \"$(dirname \"$0\")\" && pwd)\"
source $this_script_dir/../amber.sh

$AMBERHOME/bin/wrapped_progs/$<TARGET_FILE_NAME:${EXECUTABLE}> \"$@\"")

		endif()

		# install script
		install(PROGRAMS ${WRAPPER_SCRIPT} DESTINATION ${BINDIR})

		# install actual program
		install(TARGETS ${EXECUTABLE} DESTINATION ${BINDIR}/wrapped_progs)

		if(TARGET_OSX)
			# since the program is now going into a subfolder we need to change its RPATH
			# to account for this.
			set_property(TARGET ${EXECUTABLE} PROPERTY INSTALL_RPATH "@loader_path/../../${LIBDIR}")

		endif()

	endforeach()

endfunction(install_executables_wrapped)