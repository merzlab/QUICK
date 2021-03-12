# File which figures out the compiler flags to use based on the vendor and version of each compiler
#Note: must be included after OpenMPConfig, MPIConfig, and PythonConfig

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

set(OPT_FFLAGS -O2)
set(OPT_CFLAGS -O2)
set(OPT_CXXFLAGS -O2)

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


#-------------------------------------------------------------------------------
#  Now, the If Statements of Doom...
#-------------------------------------------------------------------------------

#gcc
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

    if (WARNINGS)
	add_flags(C -Wall -Wno-unused-function -Wno-unknown-pragmas)
    
	if(NOT UNUSED_WARNINGS)
	    add_flags(C -Wno-unused-variable -Wno-unused-but-set-variable)
	endif()
    
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(C -Wno-uninitialized -Wno-maybe-uninitialized)
	endif()
    endif()
    
    if(${CMAKE_C_COMPILER_VERSION} VERSION_GREATER 4.1)
	if(SSE)
	    if(TARGET_ARCH STREQUAL x86_64)
		#-mfpmath=sse is default for x86_64, no need to specific it
		list(APPEND OPT_CFLAGS "-mtune=native")
	    else() # i386 needs to be told to use sse prior to using -mfpmath=sse
		list(APPEND OPT_CFLAGS "-mtune=native -msse -mfpmath=sse")
	    endif()
	endif()
    endif()    
    
    if(DRAGONEGG)
	#check dragonegg
	check_c_compiler_flag(-fplugin=${DRAGONEGG} DRAGONEGG_C_WORKS)
	if(NOT DRAGONEGG_C_WORKS)
	    message(FATAL_ERROR "Can't use C compiler with Dragonegg.  Please fix whatever's broken.  Check CMakeOutput.log for details.")
	endif()
	
	add_flags(C -fplugin=${DRAGONEGG})
    endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    
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
	
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 4.1)
	if(SSE)
	    if(TARGET_ARCH STREQUAL x86_64)
          	#-mfpmath=sse is default for x86_64, no need to specific it
          	list(APPEND OPT_CXXFLAGS "-mtune=native")
            else() # i386 needs to be told to use sse prior to using -mfpmath=sse
          	list(APPEND OPT_CXXFLAGS "-mtune=native -msse -mfpmath=sse")
            endif()
        endif()
    endif()    
  
    if(DRAGONEGG)
	#check dragonegg
	check_cxx_compiler_flag(-fplugin=${DRAGONEGG} DRAGONEGG_CXX_WORKS)
	if(NOT DRAGONEGG_CXX_WORKS)
	    message(FATAL_ERROR "Can't use C++ compiler with Dragonegg.  Please fix whatever's broken.  Check CMakeOutput.log for details.")
	endif()
		
	add_flags(CXX -fplugin=${DRAGONEGG})
    endif()
	
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")

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
		
    if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_GREATER 4.1)
	if(SSE)
	    if(TARGET_ARCH STREQUAL x86_64)
          	#-mfpmath=sse is default for x86_64, no need to specific it
          	set(OPT_FFLAGS ${OPT_FFLAGS} -mtune=native)
            else() # i386 needs to be told to use sse prior to using -mfpmath=sse
          	set(OPT_FFLAGS ${OPT_FFLAGS} -mtune=native -msse -mfpmath=sse)
            endif()
        endif()
    endif()
	
	
    # gcc 4.1.2 does not support putting allocatable arrays in a Fortran type...
    # so unfortunately file-less prmtop support in the sander API will not work
    # in this case.
    if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_LESS 4.2)
        add_definitions(-DNO_ALLOCATABLES_IN_TYPE)
    endif()
    
    # Check dragonegg
    if(DRAGONEGG)
	#TODO: write check_fortran_compiler_flag
	#check_fortran_compiler_flag(-fplugin=${DRAGONEGG} DRAGONEGG_FORTRAN_WORKS)
	#if(NOT DRAGONEGG_FORTRAN_WORKS)
	#	message(FATAL_ERROR "Can't use Fortran compiler with Dragonegg.  Please fix whatever's broken.  Check CMakeOutput.log for details.")
	#endif()
		
	add_flags(Fortran -fplugin=${DRAGONEGG})
    endif()
endif()

#clang
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
	add_flags(C -Wall -Wno-unused-function)
	
	list(APPEND OPT_CFLAGS "-mtune=native")
	
	#if we are crosscompiling and using clang, tell CMake this
	if(CROSSCOMPILE)
		set(CMAKE_C_COMPILER_TARGET ${TARGET_TRIPLE})
	endif()  
	
	if(NOT UNINITIALIZED_WARNINGS)
		add_flags(C -Wno-sometimes-uninitialized)
	endif()
	
	if(OPENMP AND (${CMAKE_C_COMPILER_VERSION} VERSION_LESS 3.7))
		message(FATAL_ERROR "Clang versions earlier than 3.7 do not support OpenMP!  Disable it or change compilers!")
	endif()
		
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
	add_flags(CXX -Wall -Wno-unused-function)
	
	list(APPEND OPT_CXXFLAGS "-mtune=native")
	
	if(CROSSCOMPILE)
		set(CMAKE_CXX_COMPILER_TARGET ${TARGET_TRIPLE})
	endif()
	
	if(NOT UNINITIALIZED_WARNINGS)
		add_flags(CXX -Wno-sometimes-uninitialized)
	endif()
	
	if(TARGET_OSX)
		# on OS X, Python will link pytraj's extension modules to libc++, so cpptraj needs to use the same standard library
		add_flags(CXX -stdlib=libc++)
		if(NOT $ENV{CONDA_BUILD})
			set(CMAKE_OSX_DEPLOYMENT_TARGET 10.7)
		endif()
	endif()	
	
	if(OPENMP AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.7))
		message(FATAL_ERROR "Clang versions earlier than 3.7 do not support OpenMP!  Disable it or change compilers!")
	endif()
endif()

#msvc
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
	add_flags(C /D_CRT_SECURE_NO_WARNINGS /MP)
	
	set(OPT_CFLAGS "/Ox")
	set(NO_OPT_CFLAGS "/Od")
	
	set(CMAKE_C_FLAGS_DEBUG "/Zi")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	add_flags(CXX /D_CRT_SECURE_NO_WARNINGS /MP)
	
	set(OPT_CXXFLAGS "/Ox")
	set(NO_OPT_CXXFLAGS "/Od")
	
	set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
endif()

#intel
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_C_FLAGS_DEBUG "-g -debug all")
	
	set(OPT_CFLAGS -ip -O3)
		
	#  How flags get set for optimization depend on whether we have a MIC processor,
    #  the version of Intel compiler we have, and whether we are cross-compiling
    #  for multiple versions of SSE support.  The following coordinates all of this.
    #  This was done assuming that MIC and SSE are mutually exclusive and that we want
    #  SSE instructions included only when optimize = yes.  Note that use of an
    #  SSE_TYPES specification needs to be given in place of xHost not in addition to.
    #  This observed behavior is not what is reported by the Intel man pages. BPK
	
	if(SSE)
		# BPK removed section that modified O1 or O2 to be O3 if optimize was set to yes.
      	# We already begin with the O3 setting so it wasn't needed.
        # For both coptflags and foptflags, use the appropriate settings
        # for the sse flags (compiler version dependent).
        if(${CMAKE_C_COMPILER_VERSION} VERSION_GREATER 11 OR ${CMAKE_C_COMPILER_VERSION} VERSION_EQUAL 11)
			if(NOT "${SSE_TYPES}" STREQUAL "")
				list(APPEND OPT_CFLAGS "-ax${SSE_TYPES}")
			else()
				list(APPEND OPT_CFLAGS -xHost)
			endif()
		else()
			list(APPEND OPT_CFLAGS -axSTPW)
		endif()
		
	endif()
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")

	if(WIN32)
		add_flags(Fortran /D_CRT_SECURE_NO_WARNINGS)
	
		set(OPT_FFLAGS "/Ox")
		
		set(CMAKE_Fortran_FLAGS_DEBUG "/Zi")
	else()
		set(CMAKE_Fortran_FLAGS_DEBUG "-g -debug all")
		
		set(OPT_FFLAGS -ip -O3)
		
		if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_EQUAL 18)
			message(WARNING "Significant test failures were experienced with 2018 versions of Intel compilers!  Workarounds for these known problems have been implemented.  \
However, we do not recommend building Amber with icc version 18. Versions 19, 17, and 16 are much more stable.")
		endif()
			
		if(SSE)

			if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_GREATER 11 OR ${CMAKE_Fortran_COMPILER_VERSION} VERSION_EQUAL 11)
				if(NOT "${SSE_TYPES}" STREQUAL "")
					list(APPEND OPT_FFLAGS "-ax${SSE_TYPES}")
				else()
					list(APPEND OPT_FFLAGS -xHost)
				endif()
			else()
				list(APPEND OPT_FFLAGS -axSTPW)
			endif()
		endif()
		
		
		# warning flags
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
	
	set(OPT_CXXFLAGS -O3)
endif()

# PGI
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
if("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
	set(OPT_CFLAGS -O2)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
	set(OPT_CXXFLAGS -O2)
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	set(OPT_FFLAGS -fast -O3)
	set(NO_OPT_FFLAGS -O1)
	
	if(SSE)
		list(APPEND OPT_FFLAGS -fastsse)
	endif()
	
endif()

# Cray
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
if("${CMAKE_C_COMPILER_ID}" STREQUAL "Cray")

    # NOTE: In order for GNU-like defines to work (e.g.
    #       -D_FILE_OFFSET_BITS etc.) cray compilers need '-h gnu'.

    add_flags(C -h gnu)
    
    # cray compilers have equivalent of -O3 on by default
    set(OPT_CFLAGS "")
    
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")

	add_flags(CXX -h gnu)
	set(OPT_CXXFLAGS "")
	
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray")

    # Also, the fortran compile requires '-emf' to force
    # the build of module files with all-lowercase names.
    add_flags(Fortran -h gnu -emf)
    
    set(OPT_FFLAGS "")
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

#this doesn't seem to get defined automatically, at least in some situations
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_definitions(-DWIN32)
endif()

# the mother of all feature-test macros
# tells the C library to enable all functions which are specified by any standard
add_definitions(-D_GNU_SOURCE)

if(NOT DOUBLE_PRECISION)
	add_definitions(-D_REAL_) #This is read by dprec.fh, where it determines the type of precision to use
endif()

# This definition gets applied to everything, everywhere
# I think you used to be able to enable or disable netcdf, and this was the switch for it
add_definitions(-DBINTRAJ)

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
