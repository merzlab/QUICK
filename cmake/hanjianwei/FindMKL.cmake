# - Find Intel MKL
# modified for AMBER
# Find the MKL libraries
#
# NOTE: MKL_MULTI_THREADED requires the patched FindOpenMPFixed module from the Amber-MD/cmake-buildscripts repository.
# 
# Options:
#
#   MKL_STATIC        :   use static linking.  Requires linker support for the -Wl,--start-group flag.
#   MKL_MULTI_THREADED:   use multi-threading. Requires the FindOpenMP module 
# 	MKL_NEEDEXTRA	  :   Also import the "extra" MKL libraries (scalapack and cdft)
#	MKL_NEEDINCLUDES  :	  Set to true if you require mkl.h.  Since many applications don't need the header, if this is false, MKL will still count as found even if the headers aren't found.
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found.  Note that
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc (can be a multi-element list)
#   MKL_INCLUDE_DIRS     : alias for MKL_INCLUDE_DIR
#   MKL_LIBRARIES        : Libraries to link against for your configuration when using C or C++. 
#	MKL_FORTRAN_LIBRARIES: Libraries to link against when any Fortran code is being linked.  Use these everywhere if you are mixing Fortran and C/C++.
#
# It also creates some imported targets.  Some are for internal use since they have to be used with the correct linker flags.
# The ones for external use, enabled by the MKL_NEEDEXTRA variable, are:
#   
#   mkl::cdft 		- MKL cdft library for Complex Distributed fast Fourier Transforms
#	mkl::scalapack 	- MKL ScaLAPACK library for MPI LAPACK operations

include(FindPackageHandleStandardArgs)

if((NOT DEFINED MKL_HOME) OR NOT EXISTS "${MKL_HOME}")
	# try to find in environment variables..
	if(DEFINED ENV{MKL_HOME})
		set(MKL_HOME $ENV{MKL_HOME} CACHE PATH "Root folder of Math Kernel Library")
	elseif(DEFINED ENV{MKLROOT})
		set(MKL_HOME $ENV{MKLROOT} CACHE PATH "Root folder of Math Kernel Library")
	# now try default folders
	elseif(WIN32 AND EXISTS "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl")
		set(MKL_HOME "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl" CACHE PATH "Root folder of Math Kernel Library")
	elseif(EXISTS "/opt/intel/mkl")
		set(MKL_HOME "/opt/intel/mkl" CACHE PATH "Root folder of Math Kernel Library")
	else()
		message(STATUS "Unable to locate MKL_HOME for your system.  To use MKL, set MKL_HOME to point to your MKL installation location.")
		set(MKL_HOME "" CACHE PATH "Root folder of Math Kernel Library")
	endif()
endif()

# local version of import_library() that does not add to the library tracker
#usage: import_library(<library name> <library path> [include dir 1] [include dir 2]...)
function(mkl_import_library NAME PATH) #3rd arg: INCLUDE_DIRS
	
	#Try to figure out whether it is shared or static.
	get_lib_type(${PATH} LIB_TYPE)

	if("${LIB_TYPE}" STREQUAL "SHARED")
		add_library(${NAME} SHARED IMPORTED GLOBAL)
	else()
		add_library(${NAME} STATIC IMPORTED GLOBAL)
	endif()

	set_property(TARGET ${NAME} PROPERTY IMPORTED_LOCATION ${PATH})
	set_property(TARGET ${NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARGN})
	
endfunction(mkl_import_library)

######################### Headers #######################

# Find include dir
find_path(MKL_INCLUDE_DIR mkl.h HINTS ${MKL_HOME}/include)
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

# Find include directory
#  There is no include folder under linux
if(WIN32)
    find_path(INTEL_INCLUDE_DIR omp.h PATHS ${MKL_HOME}/../include)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} ${INTEL_INCLUDE_DIR})
endif()

# avoid passing "-NOTFOUND" to imported libraries
if(MKL_INCLUDE_DIR)
	set(MKL_INCLUDE_DIR_FOR_IMPLIB ${MKL_INCLUDE_DIR})
else()
	set(MKL_INCLUDE_DIR_FOR_IMPLIB "")
endif()

######################### Find libraries #######################

# Handle suffix
set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MKL_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
	if(DEFINED CMAKE_IMPORT_LIBRARY_SUFFIX)
    	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_IMPORT_LIBRARY_SUFFIX})
	else()
    	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
	endif()
endif()

# names of subdirectories in the lib folder
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(MKL_ARCHITECTURES intel64 em64t)
else()
	set(MKL_ARCHITECTURES ia32)
endif()

set(MKL_LIB_PATHS "")
set(MKL_OMP_LIB_PATHS "")# paths to look for the Intel OpenMP runtime library in

foreach(ARCH ${MKL_ARCHITECTURES})
	list(APPEND MKL_LIB_PATHS ${MKL_HOME}/lib/${ARCH})
	list(APPEND MKL_OMP_LIB_PATHS ${MKL_HOME}/../compiler/lib/${ARCH} ${MKL_HOME}/../lib/${ARCH})
endforeach()

# MKL is composed of four layers: Interface, Threading, Computational and OpenMP

######################### Interface layer #######################

# NOTE: right now it's hardcoded to use the 32-bit compatibility versions of certain libraries (lp64 instead of ilp64)

if(WIN32)
    set(MKL_INTERFACE_LIBNAMES mkl_intel mkl_intel_lp64 mkl_intel_c mkl_intel_c_lp64)
else()
    set(MKL_INTERFACE_LIBNAMES mkl_intel mkl_intel_lp64)
endif()

find_library(MKL_INTERFACE_LIBRARY NAMES ${MKL_INTERFACE_LIBNAMES} HINTS ${MKL_LIB_PATHS})

find_library(MKL_GFORTRAN_INTERFACE_LIBRARY NAMES mkl_gf mkl_gf_lp64 HINTS ${MKL_LIB_PATHS})
	
# gfortran specifically needs a seperate library
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set(MKL_FORTRAN_INTERFACE_LIBRARY MKL_GFORTRAN_INTERFACE_LIBRARY)
else()
	set(MKL_FORTRAN_INTERFACE_LIBRARY MKL_INTERFACE_LIBRARY)
endif()

if(MKL_INTERFACE_LIBRARY)
	mkl_import_library(mkl::interface ${MKL_INTERFACE_LIBRARY} ${MKL_INCLUDE_DIR_FOR_IMPLIB})
endif()

if(${MKL_FORTRAN_INTERFACE_LIBRARY})
	mkl_import_library(mkl::fortran_interface ${${MKL_FORTRAN_INTERFACE_LIBRARY}} ${MKL_INCLUDE_DIR_FOR_IMPLIB})
endif()

######################## Threading layer ########################
find_library(MKL_SEQUENTIAL_THREADING_LIBRARY mkl_sequential HINTS ${MKL_LIB_PATHS})
find_library(MKL_INTEL_THREADING_LIBRARY mkl_intel_thread HINTS ${MKL_LIB_PATHS})
find_library(MKL_GNU_THREADING_LIBRARY mkl_gnu_thread HINTS ${MKL_LIB_PATHS})
find_library(MKL_PGI_THREADING_LIBRARY mkl_pgi_thread HINTS ${MKL_LIB_PATHS})


if(MKL_MULTI_THREADED)
	
	# this might not be the best when mixing different compilers, but I'm not sure there IS a correct action to take in that case.
	if("${CMAKE_C_COMPILER_ID}" STREQUAL GNU)
		set(MKL_THREADING_LIBRARY MKL_GNU_THREADING_LIBRARY)
	elseif("${CMAKE_C_COMPILER_ID}" STREQUAL PGI)	
		set(MKL_THREADING_LIBRARY MKL_PGI_THREADING_LIBRARY)
    else()
    	set(MKL_THREADING_LIBRARY MKL_INTEL_THREADING_LIBRARY)
    endif()
    
else()
    set(MKL_THREADING_LIBRARY MKL_SEQUENTIAL_THREADING_LIBRARY)
endif()

if(${MKL_THREADING_LIBRARY})
	mkl_import_library(mkl::threading "${${MKL_THREADING_LIBRARY}}" ${MKL_INCLUDE_DIR_FOR_IMPLIB})
endif()

####################### Computational layer #####################
find_library(MKL_CORE_LIBRARY mkl_core HINTS ${MKL_LIB_PATHS})

if(MKL_CORE_LIBRARY)
	mkl_import_library(mkl::core "${MKL_CORE_LIBRARY}" ${MKL_INCLUDE_DIR_FOR_IMPLIB})
endif()

if(MKL_NEEDEXTRA)
	find_library(MKL_FFT_LIBRARY mkl_cdft_core HINTS ${MKL_LIB_PATHS})
	find_library(MKL_SCALAPACK_LIBRARY mkl_scalapack_core mkl_scalapack_lp64 HINTS ${MKL_LIB_PATHS})
	
	if(MKL_FFT_LIBRARY)
		mkl_import_library(mkl::cdft "${MKL_FFT_LIBRARY}" ${MKL_INCLUDE_DIR_FOR_IMPLIB})
	endif()
	if(MKL_SCALAPACK_LIBRARY)
		mkl_import_library(mkl::scalapack "${MKL_SCALAPACK_LIBRARY}" ${MKL_INCLUDE_DIR_FOR_IMPLIB})
	endif()
endif()

############################ OpenMP Library ##########################

if(MKL_MULTI_THREADED)
	find_package(OpenMP)
	
	# NOTE: we don't want to link against the imported targets, because that would apply OpenMP compile flags to anything linked to MKL
	set(MKL_OMP_LIBRARY ${OpenMP_C_OPTIONS} ${OpenMP_C_LIBRARIES})
else()
	set(MKL_OMP_LIBRARY "")
endif()
    
   ############################ Link Options ##########################


if(MKL_STATIC)
	# figure out how to link the static libraries
	check_linker_flag(-Wl,--start-group C SUPPORTS_LIB_GROUPS)
	
	if(NOT SUPPORTS_LIB_GROUPS)
		message(WARNING "Your linker does not support library grouping.  MKL cannot be linked statically on this platform.")
		set(LIB_LIST_PREFIX LIB_LIST_PREFIX-NOTFOUND)
		set(LIB_LIST_SUFFIX LIB_LIST_SUFFIX-NOTFOUND)
	else()
		set(LIB_LIST_PREFIX -Wl,--start-group)
		set(LIB_LIST_SUFFIX -Wl,--end-group)
	endif()
	
else()
	check_linker_flag(-Wl,--no-as-needed C SUPPORTS_NO_AS_NEEDED)
	
	if(SUPPORTS_NO_AS_NEEDED)
		set(LIB_LIST_PREFIX -Wl,--no-as-needed)
	else()
		# we *hope* that the linker doesn't do as-needed linking at all and thus the flag is not necessary
		set(LIB_LIST_PREFIX "")
	endif()
	
	set(LIB_LIST_SUFFIX "")
endif()

# Library names to pass to FPHSA
set(MKL_NEEDED_LIBNAMES MKL_INTERFACE_LIBRARY ${MKL_FORTRAN_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} MKL_CORE_LIBRARY)
if(MKL_NEEDINCLUDES)
	list(APPEND MKL_NEEDED_LIBNAMES MKL_INCLUDE_DIR)
endif()
if(MKL_MULTI_THREADED)
	list(APPEND MKL_NEEDED_LIBNAMES OpenMP_C_FOUND)
endif()

# fix MKL_INTERFACE_LIBRARY appearing twice if the fortran interface library is the same as the normal one
list(REMOVE_DUPLICATES MKL_NEEDED_LIBNAMES)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

############################ Extra Libraries ##########################


# link libdl if it exists
find_library(LIBDL_LIBRARY
	NAMES dl)
if(LIBDL_LIBRARY)
	try_link_library(HAVE_LIBDL
		LANGUAGE C
		FUNCTION dlopen
		LIBRARIES ${LIBDL_LIBRARY})
else()
	set(HAVE_LIBDL FALSE)
endif()


if(HAVE_LIBDL)
	set(MKL_LIBDL ${LIBDL_LIBRARY})
else()
	set(MKL_LIBDL "")
endif()

# Link pthread if it exists
find_package(Threads)

if(CMAKE_THREAD_LIBS_INIT)
	set(MKL_PTHREAD_LIB Threads::Threads)
else()
	set(MKL_PTHREAD_LIB "")
endif()

# link CMath
find_package(CMath)
list(APPEND MKL_NEEDED_LIBNAMES CMath_FOUND)

# Build the final library lists    
set(MKL_LIBRARIES ${LIB_LIST_PREFIX} mkl::core mkl::threading mkl::interface ${LIB_LIST_SUFFIX} ${MKL_LIBDL} ${MKL_PTHREAD_LIB} ${MKL_OMP_LIBRARY} C::Math)
set(MKL_FORTRAN_LIBRARIES ${LIB_LIST_PREFIX} mkl::core mkl::threading mkl::fortran_interface ${LIB_LIST_SUFFIX} ${MKL_LIBDL} ${MKL_PTHREAD_LIB} ${MKL_OMP_LIBRARY} C::Math)

cmake_policy(SET CMP0060 NEW)

# Check that it works
try_link_library(MKL_WORKS
	LANGUAGE C
	FUNCTION sgemm
	LIBRARIES ${MKL_LIBRARIES})

list(APPEND MKL_NEEDED_LIBNAMES MKL_WORKS)

if(CMAKE_Fortran_COMPILER_LOADED)
	try_link_library(MKL_FORTRAN_WORKS
		LANGUAGE Fortran
		FUNCTION sgemm
		LIBRARIES ${MKL_FORTRAN_LIBRARIES})
	list(APPEND MKL_NEEDED_LIBNAMES MKL_FORTRAN_WORKS)
endif()
		
find_package_handle_standard_args(MKL DEFAULT_MSG ${MKL_NEEDED_LIBNAMES})
