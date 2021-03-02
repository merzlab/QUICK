# - Find NetCDF
# Find the native NetCDF includes and library
# Modified for AMBER
#
#  NetCDF_INCLUDES    - where to find netcdf.h, etc
#  NetCDF_LIBRARIES   - Link these libraries when using NetCDF
#  NetCDF_FOUND       - True if NetCDF found including required interfaces (see below)
#
# Your package can ask for certain COMPONENTS (in addition to the C library):
#
#  CXX         - require the C++ interface and link the C++ library
#  F77         - require the F77 interface and link the fortran library
#  F90         - require the F90 interface and link the fortran library
#
# The following are not for general use and are included in
# NETCDF_LIBRARIES if the corresponding option above is set.
#
#  NetCDF_LIBRARIES_C    - Just the C interface
#  NetCDF_LIBRARIES_CXX  - C++ interface, if available
#  NetCDF_LIBRARIES_F77  - Fortran 77 interface, if available
#  NetCDF_LIBRARIES_F90  - Fortran 90 interface, if available
#
# Normal usage would be:
#  find_package (NETCDF REQUIRED COMPONENTS F90)
#  target_link_libraries (uses_f90_interface ${NETCDF_LIBRARIES})
#  target_link_libraries (only_uses_c_interface ${NETCDF_LIBRARIES_C})

if (NetCDF_INCLUDES AND NetCDF_LIBRARIES)
  # Already in cache, be silent
  set (NetCDF_FIND_QUIETLY TRUE)
endif (NetCDF_INCLUDES AND NetCDF_LIBRARIES)

find_path (NetCDF_INCLUDES netcdf.h)
find_library(NetCDF_LIBRARIES_C NAMES netcdf)

if(NetCDF_LIBRARIES_C)
	get_filename_component(NetCDF_LIBPATH "${NetCDF_LIBRARIES_C}" DIRECTORY)
	
	check_library_exists(netcdf nc_strerror ${NetCDF_LIBPATH} NetCDF_C_WORKS)
	
	if(NOT NetCDF_C_WORKS)
		# cause the test to be repeated on failure so that the user can debug it 
		unset(NetCDF_C_WORKS CACHE)
	endif()
endif()

mark_as_advanced(NetCDF_LIBRARIES_C)

set(NetCDF_LIBS "${NetCDF_LIBRARIES_C}")

get_filename_component(NetCDF_lib_dirs "${NetCDF_LIBRARIES_C}" PATH)

macro (NetCDF_check_interface lang header libs)
	find_path (NetCDF_INCLUDES_${lang} NAMES ${header} HINTS "${NetCDF_INCLUDES}" NO_DEFAULT_PATH)
  
	find_library (NetCDF_LIBRARIES_${lang} NAMES ${libs} HINTS "${NetCDF_lib_dirs}")

	mark_as_advanced (NetCDF_LIBRARIES_${lang})
		
	if (NetCDF_INCLUDES_${lang} AND NetCDF_LIBRARIES_${lang})
		list (INSERT NetCDF_LIBS 0 ${NetCDF_LIBRARIES_${lang}}) # prepend so that -lnetcdf is last
		set (NetCDF_${lang}_FOUND TRUE)
	else()
		set (NetCDF_${lang}_FOUND FALSE)
		message (STATUS "Failed to find NetCDF interface for ${lang} (NetCDF_INCLUDES_${lang} = ${NetCDF_INCLUDES_${lang}}, NetCDF_LIBRARIES_${lang} = ${NetCDF_LIBRARIES_${lang}})")
	endif()
endmacro (NetCDF_check_interface)


if("${NetCDF_FIND_COMPONENTS}" MATCHES CXX)
	NetCDF_check_interface (CXX netcdfcpp.h netcdf_c++)
endif()

if("${NetCDF_FIND_COMPONENTS}" MATCHES F77)
	NetCDF_check_interface (F77 netcdf.inc  netcdff)
endif()

if("${NetCDF_FIND_COMPONENTS}" MATCHES F90)
	NetCDF_check_interface (F90 netcdf.mod  netcdff)
	
	if(NetCDF_F90_FOUND)
		# we must check that netcdf.mod is compatible with the Fortran compiler in use
		# --------------------------------------------------------------------
		set(CMAKE_REQUIRED_INCLUDES ${NetCDF_INCLUDES_F90})
		set(CMAKE_REQUIRED_LIBRARIES ${NetCDF_LIBS})
		# Test NetCDF Fortran
		check_fortran_source_runs(
"program testf
  use netcdf
  write(6,*) nf90_strerror(0)
  write(6,*) 'testing a Fortran program'
end program testf"
	NetCDF_F90_WORKS)
		unset(CMAKE_REQUIRED_INCLUDES)
		unset(CMAKE_REQUIRED_LIBRARIES)
	
		if(NOT NetCDF_F90_WORKS)
			#force the compile test to be repeated next configure
			unset(NetCDF_F90_WORKS CACHE)
			
			# print an expository message
			message(STATUS "NetCDF's Fortran 90 interface was found on your system, but it wasn't possible to compile programs using it.  The module file in use is \
\"${NetCDF_INCLUDES_F90}/netcdf.mod\", and the library is located at \"${NetCDF_LIBRARIES_F90}\"  The most common cause of this error is that the module was \
created by a different compiler or compiler version than what's currently in use.  Check CMakeError.log for more details.")
	
			set(NetCDF_F90_FOUND FALSE)
			
		endif()
	endif()
endif()

set (NetCDF_LIBRARIES "${NetCDF_LIBS}")

# handle the QUIETLY and REQUIRED arguments and set NetCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF HANDLE_COMPONENTS REQUIRED_VARS NetCDF_C_WORKS NetCDF_LIBRARIES NetCDF_INCLUDES)

mark_as_advanced (NetCDF_LIBRARIES NetCDF_INCLUDES)
