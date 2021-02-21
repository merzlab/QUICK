function(check_mkl)
	SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
	SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
    set(MKLLIBPATH $ENV{MKLROOT}/lib/intel64)
	find_library(MKL_INTEL_LP64 
				NAMES mkl_intel_lp64
				PATHS ${MKLLIBPATH})
    if(NOT MKL_INTEL_LP64)
        message(WARNING "libmkl_intel_lp64.so cannot be found. Inbuilt diagonalizer will be used.")
        return()
    endif()

    find_library(MKL_INTEL_THREAD 
				NAMES mkl_intel_thread
				PATHS ${MKLLIBPATH})
    if(NOT MKL_INTEL_THREAD)
        message(WARNING "libmkl_intel_thread.so cannot be found. Inbuilt diagonalizer will be used.")
        return()
    endif()

    find_library(MKL_CORE 
				NAMES mkl_core
				PATHS ${MKLLIBPATH})
    if(NOT MKL_CORE)
        message(WARNING "libmkl_core.so cannot be found. Inbuilt diagonalizer will be used.")
        return()
    endif()

    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DMKL -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"
		PARENT_SCOPE)
	message(STATUS "Matrix diagonalizer from MKL will be used.")
endfunction(check_mkl)

function(check_lapack)
	SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
	SET(BLASLIBPATH $ENV{BLASROOT}/lib)
    find_library(OPENBLAS 
				NAMES openblas
				PATHS ${BLASLIBPATH})
    if(NOT OPENBLAS)
        message(WARNING "libopenblas.so cannot be found. Inbuilt diagonalizer will be used.")
        return()
    endif()
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DLAPACK -lopenblas" PARENT_SCOPE)
	message(STATUS "Matrix diagonalizer from LAPACK will be used.")
endfunction(check_lapack)

if(SHARED)
	message(STATUS "Build shared libs")
	set(BUILD_SHARED_LIBS TRUE)
endif()

if(DEBUG)
	message(STATUS "Build type is debug")
	set(CMAKE_BUILD_TYPE "Debug")
	add_definitions(-DDEBUG)
endif()

add_compile_options($<$<CONFIG:Debug>:-g> $<$<CONFIG:Debug>:-DDEBUG>)

set(OPT_FFLAGS "-O2")
set(OPT_CFLAGS "-O2")
set(OPT_CXXFLAGS "-O2")
set(CMAKE_Fortran_FLAGS_DEBUG "-O0")
set(CMAKE_C_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")

if(${COMPILER} STREQUAL GNU)
	add_definitions(-DGNU)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mtune=native -ffree-form -cpp")  
elseif(${COMPILER} STREQUAL INTEL)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ip -cpp -diag-disable 8291")
	set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -traceback")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -traceback")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -traceback")
endif()

if(MKL)
	check_mkl()
endif()

if(LAPACK)
	check_lapack()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include(QuickCompilerConfig)
include(CopyTarget)
include(ConfigModuleDirs)


