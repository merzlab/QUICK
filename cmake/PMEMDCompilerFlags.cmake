#File which sets special compiler flags for PMEMD
#Often little configuration things are done in the subdir CMakeLists, but since the logic is so complicated I wanted to do this here in the root folder


# PMEMD config options
option(MIC_KC "Build with optimizations for Intel Knight's Landing processor." FALSE)
option(PMEMD_SPDP "Build pmemd's midpoint code and mic_kc optimzations using SPDP precision." FALSE)
option(PMEMD_OMP_MPI "Build the ultimate in parallelization: pmemd.OMP.MPI" FALSE)
#-------------------------------------------------------------------------------
#  Set default flags.  
#-------------------------------------------------------------------------------

# definitions for all pmemd versions
set(PMEMD_DEFINITIONS EMIL) # EMIL is always defined as far as I can tell

# Compile flags applied to all C and Fortran source files, respectively
set(PMEMD_CFLAGS "")
set(PMEMD_FFLAGS "")

# MIC flags, shared between C and Fortran
set(PMEMD_MIC_FLAGS "")

# Definitions and flags applied to pmemd.MPI
set(PMEMD_MPI_DEFINITIONS "")
set(PMEMD_MPI_FLAGS "")

# Definitions and flags applied to pmemd.MPI.OMP
set(PMEMD_OMP_DEFINITIONS "")
set(PMEMD_OMP_FLAGS "")

#-------------------------------------------------------------------------------
#  CUDA precisions
#  For each value in theis variable, the CUDA code will be built again with use_<value> defined,
#  and a new pmemd.<value> executable will be created
#-------------------------------------------------------------------------------

set(PMEMD_CUDA_PRECISIONS SPFP DPFP)

#precision of pmemd which gets installed as pmemd.cuda
set(PMEMD_DEFAULT_PRECISION SPFP)

#-------------------------------------------------------------------------------
# Optimization flag configuration  
#-------------------------------------------------------------------------------

if(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")

      
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	
		set(PMEMD_CFLAGS -O3 -mdynamic-no-pic -no-prec-div -ipo -xHost)

	else()

		set(PMEMD_CFLAGS -ip -O3 -no-prec-div)
		
		if(SSE)
			if(NOT "${SSE_TYPES}" STREQUAL "")
				list(APPEND PMEMD_CFLAGS "-ax${SSE_TYPES}")
			else()
				list(APPEND PMEMD_CFLAGS -xHost)
			endif()
		endif()

	endif()
	
	
else()
	#use regular compiler optimization flags
	set(PMEMD_CFLAGS ${OPT_CFLAGS})
endif()

#Fortran
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#this tree mirrors the C tree very closely, with only minor differences
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	
		set(PMEMD_FFLAGS -O3 -mdynamic-no-pic -no-prec-div -ipo -xHost)
		
		list(APPEND PMEMD_FFLAGS -ipo)
   
	else()
		set(PMEMD_FFLAGS -ip -O3 -no-prec-div)


		if(SSE)
			if(NOT "${SSE_TYPES}" STREQUAL "")
				list(APPEND PMEMD_FFLAGS "-ax${SSE_TYPES}")
			else()
				list(APPEND PMEMD_FFLAGS -xHost)
			endif()
		endif()

	endif()
	
else()
	#use regular compiler flags
	set(PMEMD_FFLAGS ${OPT_FFLAGS})
endif()

	
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_Fortran_COMPILER_VERSION}" VERSION_EQUAL 5)	
	# compile pmemd prmtop_dat at lower optimization for buggy gnu 5.x: see bug 303.
	set(PMEMD_GNU_BUG_303 TRUE) 
else()
	set(PMEMD_GNU_BUG_303 FALSE)
endif()


#-------------------------------------------------------------------------------
#  OpenMP configuration
#-------------------------------------------------------------------------------

if(OPENMP)

	list(APPEND PMEMD_OMP_DEFINITIONS _OPENMP_ MP_VEC)

	# extra openmp flags for ICC
	if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
		if(NOT "${CMAKE_Fortran_COMPILER_VERSION}" VERSION_LESS 13)
			if("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_LESS 16)
				# Create a combined flags set that is safe for targets of both languages
				list(APPEND PMEMD_OMP_FLAGS $<$<COMPILE_LANGUAGE:Fortran>:-opt-streaming-cache-evict=0 -fp-model fast=2 -align array64byte>)
			else()
				list(APPEND PMEMD_OMP_FLAGS $<$<COMPILE_LANGUAGE:Fortran>:-qopt-streaming-cache-evict=0 -fp-model fast=2 -align array64byte>)
			endif()
		endif()
	endif()
endif()

#-------------------------------------------------------------------------------
#  Midpoint precision
#-------------------------------------------------------------------------------

if(PMEMD_SPDP)
	list(APPEND PMEMD_DEFINITIONS pmemd_SPDP MIDPOINT_SPDP) 
else()
	list(APPEND PMEMD_DEFINITIONS pmemd_DPDP)
endif()

#-------------------------------------------------------------------------------
#  Finalize flags
#-------------------------------------------------------------------------------

# disable optimization flags if needed
if(NOT OPTIMIZE)
	set(PMEMD_CFLAGS ${NO_OPT_CFLAGS})
	set(PMEMD_FFLAGS ${NO_OPT_FFLAGS})
endif()

# create non-list versions for PROPERTY COMPILE_FLAGS
list_to_space_separated(PMEMD_CFLAGS_SPC ${PMEMD_CFLAGS})
list_to_space_separated(PMEMD_FFLAGS_SPC ${PMEMD_FFLAGS})
list_to_space_separated(PMEMD_NO_OPT_CFLAGS_SPC ${PMEMD_NO_OPT_CFLAGS})

#-------------------------------------------------------------------------------
# MIC flags
#-------------------------------------------------------------------------------

if(MIC_KL)

	# check compilers and version
	foreach(LANGUAGE C CXX Fortran)
		if(NOT "${CMAKE_${LANGUAGE}_COMPILER_ID}" STREQUAL "Intel")
			message(FATAL_ERROR "All Intel compilers must be used when compiling for MIC")
		endif()

		if(${CMAKE_${LANGUAGE}_COMPILER_VERSION} VERSION_LESS 12)
			message(FATAL_ERROR "Building for MIC requires Intel Compiler Suite v12 or later.")
		endif()

	endforeach()

	if(MPI AND OPENMP)

		list(APPEND PMEMD_MIC_FLAGS -DMIC2)
		
		if(${CMAKE_C_COMPILER_VERSION} VERSION_GREATER 12 AND ${CMAKE_C_COMPILER_VERSION} VERSION_LESS 16)
			list(APPEND PMEMD_MIC_FLAGS -openmp-simd)
		else()
			list(APPEND PMEMD_MIC_FLAGS -qopenmp-simd)
		endif()
		
		if(PMEMD_SPDP)
			list(APPEND PMEMD_MIC_FLAGS -Dfaster_MIC2) 
			
			if(NOT mkl_ENABLED)
				message(FATAL_ERROR "Cannot use MIC2 optimizations without Intel MPI, Intel OpenMP, and Intel MKL on.  Please enable it, or turn off PMEMD_SPDP!")
			endif()
		endif()
	else()
		message(FATAL_ERROR "Cannot use MIC2 optimizations without Intel MPI & OpenMP on.  Please pass -DOPENMP=TRUE -DMPI=TRUE and provide an intel MPI library.")
	endif()
	

endif()


#-------------------------------------------------------------------------------
#  CUDA configuration
#-------------------------------------------------------------------------------

option(GTI "Use GTI version of pmemd.cuda instead of AFE version" TRUE)

if(CUDA)
	set(PMEMD_NVCC_FLAGS -use_fast_math -O3)
	
	set(PMEMD_CUDA_DEFINES -DCUDA)
	
	if(GTI)
		list(APPEND PMEMD_CUDA_DEFINES -DGTI)
		
		message(STATUS "Building the GTI version of pmemd.cuda")
	else()
		message(STATUS "Building the AFE version of pmemd.cuda")
	endif()

	list(APPEND PMEMD_NVCC_FLAGS --std c++11) 

	if(MPI)
		list(APPEND PMEMD_NVCC_FLAGS -DMPICH_IGNORE_CXX_SEEK)
	endif()


	option(NCCL "Use NCCL for inter-GPU communications." FALSE)
	if(NCCL AND NOT nccl_ENABLED)
		message(FATAL_ERROR "NCCL is selected for inter-GPU communications but was not found.")
	endif()

	if(NCCL)
		list(APPEND PMEMD_CUDA_DEFINES -DNCCL)
	endif()


endif()



#-------------------------------------------------------------------------------
#  MKL configuration
#-------------------------------------------------------------------------------

# tell PMEMD to use MKL if it's installed
if(mkl_ENABLED)
	list(APPEND PMEMD_DEFINITIONS FFTW_FFT MKL_FFTW_FFT)
else()
	list(APPEND PMEMD_DEFINITIONS PUBFFT)
endif()
