# File which figures out the compiler flags to use based on the vendor and version of each compiler
# Note: must be included after OpenMPConfig, MPIConfig, and PythonConfig

#-------------------------------------------------------------------------------
#  Handle CMake fortran compiler version issue
#  See https://cmake.org/Bug/view.php?id=15372
#-------------------------------------------------------------------------------
	
if(CMAKE_Fortran_COMPILER_LOADED AND "${CMAKE_Fortran_COMPILER_VERSION}" STREQUAL "")

	set(CMAKE_Fortran_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION} CACHE STRING "Fortran compiler version.  May not be autodetected correctly on older CMake versions, fix this if it's wrong." FORCE)
	message(FATAL_ERROR "Your CMake is too old to properly detect the Fortran compiler version.  It is assumed to be the same as your C compiler version, ${CMAKE_C_COMPILER_VERSION}. If this is not correct, pass -DCMAKE_Fortran_COMPILER_VERSION=<correct version> to cmake.  If it is correct,just run the configuration again.")
	
endif()
	

# create linker flags
# On Windows undefined symbols in shared libraries produce errors.
# This makes them do that on Linux too.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NO_UNDEFINED_FLAG}")



#-------------------------------------------------------------------------------
#  Set default flags
#-------------------------------------------------------------------------------

set(NO_OPT_FFLAGS -O0)
set(NO_OPT_CFLAGS -O0)
set(NO_OPT_CXXFLAGS -O0)

set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_Fortran_FLAGS_DEBUG "-g")

#blank cmake's default optimization flags, we can't use these because not everything should be built optimized.
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_Fortran_FLAGS_RELEASE "")

#a macro to make things a little cleaner
#NOTE: we can't use add_compile_options because that will apply to all languages
macro(add_flags LANGUAGE) # FLAGS...
	foreach(FLAG ${ARGN})
		set(CMAKE_${LANGUAGE}_FLAGS "${CMAKE_${LANGUAGE}_FLAGS} ${FLAG}")
	endforeach()
endmacro(add_flags)


#gnu
#-------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

    set(OPT_CFLAGS -O2 -mtune=native)

    if (WARNINGS)
	add_flags(C -Wall -Wno-unused-function -Wno-unknown-pragmas)
    
	if(NOT UNUSED_WARNINGS)
	    add_flags(C -Wno-unused-variable -Wno-unused-but-set-variable)
	endif()
    
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(C -Wno-uninitialized -Wno-maybe-uninitialized)
	endif()
    endif()
    
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    
    set(OPT_CXXFLAGS -O2 -mtune=native)

    if (WARNINGS)
	add_flags(CXX -Wall -Wno-unused-function -Wno-unknown-pragmas)
    
	# Kill it!  Kill it with fire!
	check_cxx_compiler_flag(-Wno-unused-local-typedefs SUPPORTS_WNO_UNUSED_LOCAL_TYPEDEFS)

	if(SUPPORTS_WNO_UNUSED_LOCAL_TYPEDEFS)
	    add_flags(CXX -Wno-unused-local-typedefs)
	endif()
	
	if(NOT UNUSED_WARNINGS)
	    add_flags(CXX -Wno-unused-variable -Wno-unused-but-set-variable)
	endif()
	
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(CXX -Wno-uninitialized -Wno-maybe-uninitialized)
	endif()
    endif()
	
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")

    set(OPT_FFLAGS -O2 -mtune=native)

    add_flags(Fortran -ffree-line-length-none)

    if (WARNINGS)
	add_flags(Fortran -Wall -Wno-tabs -Wno-unused-function -ffree-line-length-none)
    
	if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_GREATER 4.1)	
	    add_flags(Fortran -Wno-unused-dummy-argument)
	endif()	
    
	if(NOT UNUSED_WARNINGS)
	    add_flags(Fortran -Wno-unused-variable)
	endif()
		
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(Fortran -Wno-maybe-uninitialized)
	endif()	
    endif()
		
endif()

#intel
#-------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")

    set(CMAKE_C_FLAGS_DEBUG "-g -debug all")
    
    set(OPT_CFLAGS -ip -O2 -xHost)
		
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
    
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -debug all")
	
    set(OPT_FFLAGS -ip -O2 -xHost)
	
    if(WARNINGS)
	add_flags(Fortran "-warn all" "-warn nounused")
	    
	option(IFORT_CHECK_INTERFACES "If enabled and Intel Fortran is in use, then ifort will check that types passed to functions are the correct ones, and produce warnings or errors for mismatches." FALSE)
		
	if(NOT IFORT_CHECK_INTERFACES)
			
	    # disable errors from type mismatches
	    add_flags(Fortran -warn nointerfaces)
	endif()
    endif()		
		
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

    set(CMAKE_CXX_FLAGS_DEBUG "-g -debug all")
	
    set(OPT_CXXFLAGS -ip -O2 -xHost)

endif()

#-------------------------------------------------------------------------------
#  Add some non-compiler-dependent items
#-------------------------------------------------------------------------------
if(LARGE_FILE_SUPPORT)
	if(CMAKE_SIZEOF_VOID_P LESS 8)
		add_definitions(-D_FILE_OFFSET_BITS=64)
	endif()
endif()

check_symbol_exists(mkstemp stdlib.h HAVE_MKSTEMP)
if(HAVE_MKSTEMP)
    add_definitions(-DUSE_MKSTEMP)
endif()

#-------------------------------------------------------------------------------
#  finalize the flags
#-------------------------------------------------------------------------------

#put the opt cxxflags into the CUDA flags
foreach(FLAG ${OPT_CXXFLAGS})
	list(APPEND HOST_NVCC_FLAGS ${FLAG})
endforeach()


# disable optimization flags if optimization is disabled
if(NOT OPTIMIZE)
	set(OPT_FFLAGS ${NO_OPT_FFLAGS})
	set(OPT_CFLAGS ${NO_OPT_CFLAGS})
	set(OPT_CXXFLAGS ${NO_OPT_CXXFLAGS})
endif()

#create space-separated versions of each flag set for use in PROPERTY COMPILE_FLAGS
list_to_space_separated(OPT_FFLAGS_SPC ${OPT_FFLAGS})
list_to_space_separated(OPT_CFLAGS_SPC ${OPT_CFLAGS})
list_to_space_separated(OPT_CXXFLAGS_SPC ${OPT_CXXFLAGS})

list_to_space_separated(NO_OPT_FFLAGS_SPC ${NO_OPT_FFLAGS})
list_to_space_separated(NO_OPT_CFLAGS_SPC ${NO_OPT_CFLAGS})
list_to_space_separated(NO_OPT_CXXFLAGS_SPC ${NO_OPT_CXXFLAGS})


# When a library links to an imported library with interface include directories, CMake uses the -isystem flag to include  those directories
# Unfortunately, this seems to completely not work with Fortran, so we disable it.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE) 

# Another issue is that CMake removes certain directories such as /usr/include if they
# are manually applied to the search path, since apparently manually including them
# can break some C/C++ toolchains (https://gitlab.kitware.com/cmake/cmake/issues/17966).
# However it also does this for Fortran and that breaks the ability to include Fortran
# modules found in system directories like /usr/include.
unset(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES)
