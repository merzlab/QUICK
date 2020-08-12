# File to figure out which package types are available on the system, and to handle the user's choice of package
# Also figures out the correct value for CPACK_GENERATOR

# included by Packaging.cmake

# Figure out possible package types
# --------------------------------------------------------------------

set(AVAILABLE_PACKAGE_TYPES ARCHIVE)  # Archive is built into CMake

set(ARCHIVE_FORMAT TBZ2 CACHE STRING "Format of archive to build if PACKAGE_TYPE is set to ARCHIVE.  Valid values: TBZ2, TGZ, ZIP, 7Z, TXZ, STGZ")
validate_configuration_enum(ARCHIVE_FORMAT TBZ2 TGZ ZIP 7Z TXZ STGZ)



if(HOST_WINDOWS OR TARGET_WINDOWS) # NOTE: NSIS works on Linux, so you can actually make installers when cross-compiling

	# note that this value is NOT used by CPackNSIS. It will *only* search the PATH, not any CMake variables.
	get_filename_component(MAKENSIS makensis PROGRAM)
	
	if(EXISTS "${MAKENSIS}")
		list(APPEND AVAILABLE_PACKAGE_TYPES WINDOWS_INSTALLER)
	else()
		message(STATUS "Cannot build Windows installers because NSIS is not installed, or is not on the PATH")
	endif()
	
endif()
	
if(HOST_OSX)
	list(APPEND AVAILABLE_PACKAGE_TYPES BUNDLE)
endif()

if(HOST_LINUX)

	list(APPEND AVAILABLE_PACKAGE_TYPES DEB) # Debian packages are compressed by CMake itself, so they're available on all distros
	
 	find_program(RPMBUILD_EXECUTABLE rpmbuild) # This check IS the same as the one used by CPackRPM
 	
  	if(RPMBUILD_EXECUTABLE)
  		list(APPEND AVAILABLE_PACKAGE_TYPES RPM)
  	endif()
  	
endif()
  		
 # handle PACKAGE_TYPE option
 # --------------------------------------------------------------------
set(PACKAGE_TYPE "ARCHIVE" CACHE STRING "Type of package to build when the package target is run.  See the wiki page on creating packages for details.")
 
list_contains(PACKAGE_TYPE_VALID "${PACKAGE_TYPE}" ${AVAILABLE_PACKAGE_TYPES})
list_to_space_separated(AVAILABLE_PACKAGE_TYPES_SPC ${AVAILABLE_PACKAGE_TYPES})

if(NOT PACKAGE_TYPE_VALID)
	message(FATAL_ERROR "Invalid PACKAGE_TYPE value \"${PACKAGE_TYPE}\".  Please set it to one of the following supported options for this platform: ${AVAILABLE_PACKAGE_TYPES_SPC}") 
endif()

# Now set CPACK_GENERATOR
# --------------------------------------------------------------------

if("${PACKAGE_TYPE}" STREQUAL "ARCHIVE")
	
	# ARCHIVE_FORMAT's values are also the names of the relevant CPack generators
	set(CPACK_GENERATOR ${ARCHIVE_FORMAT})
endif()

if("${PACKAGE_TYPE}" STREQUAL "WINDOWS_INSTALLER")
	set(CPACK_GENERATOR NSIS)
endif()

if("${PACKAGE_TYPE}" STREQUAL "BUNDLE")
	set(CPACK_GENERATOR Bundle)
endif()

if("${PACKAGE_TYPE}" STREQUAL "DEB" OR "${PACKAGE_TYPE}" STREQUAL "RPM")
	set(CPACK_GENERATOR ${PACKAGE_TYPE})
endif()
