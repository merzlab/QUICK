cmake_minimum_required(VERSION 3.9.0) # version 3.9.0 needed for FindMPI and FindOpenMP versions which provide imported targets

project(quick LANGUAGES NONE VERSION 20.03)


if(NOT INSIDE_AMBER)

	# initialization and include paths
	#-------------------------------------------------------------
	include(cmake/AmberBuildSystemInit.cmake NO_POLICY_SCOPE)

	#now enable the languages
	enable_language(C CXX Fortran)

	include(AmberBuildSystem2ndInit)
	
	include(MPIConfig)
	include(CudaConfig)

	set(LINALG_LIBS_REQUIRED TRUE)
	set(NEEDED_3RDPARTY_TOOLS blas lapack mkl)
	set(REQUIRED_3RDPARTY_TOOLS )
	set(BUNDLED_3RDPARTY_TOOLS blas lapack)
	include(3rdPartyTools)
	
	include(CompilerFlags)

	
	#CPack setup
	# --------------------------------------------------------------------
	set(PACKAGE_NAME "QUICK")
	set(PACKAGE_FILENAME "QUICK")

	set(BUNDLE_IDENTIFIER org.merzlab.quick)
	set(BUNDLE_SIGNATURE QUIK)
	include(Packaging)

endif()

option(NOF "Disables the compilation of QUICK's time consuming f functions in the ERI code of cuda version. Not recommended for production." FALSE)

# C/Fortran compiler flags
# --------------------------------------------------------------------

# add -DDEBUG in debug configuration
set_property(DIRECTORY . PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)


if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -traceback")
endif()


if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")
	add_definitions(-DGNU)
endif()
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -diag-disable 8291")
	set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -traceback")
endif()

# CUDA compiler flags
# --------------------------------------------------------------------

# cancel Amber arch flags, because quick only supports SM6.0 and lower
set(CUDA_NVCC_FLAGS "")

#SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX
set(SM75FLAGS -gencode arch=compute_75,code=sm_75)        
#SM7.0 = V100 and Volta Geforce / GTX Ampere?
set(SM70FLAGS -gencode arch=compute_70,code=sm_70)
#SM6.0 = GP100 / P100 = DGX-1
set(SM60FLAGS -gencode arch=compute_60,code=sm_60)

set(QUICK_USER_ARCH "" CACHE STRING "Specify QUICK gpu architecture. Applicable for cuda and cudampi versions only. If empty, QUICK will be compiled for several architectures based on the CUDA toolkit version.")

if("${QUICK_USER_ARCH}" STREQUAL "")
	# build for all supported CUDA versions
	if(${CUDA_VERSION} VERSION_EQUAL 8.0)
		message(STATUS "Configuring QUICK for SM6.0")
		list(APPEND CUDA_NVCC_FLAGS ${SM60FLAGS})
	  	
	elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0) AND (${CUDA_VERSION} VERSION_LESS 10.0)) 
		message(STATUS "Configuring QUICK for SM6.0 and SM7.0")
		list(APPEND CUDA_NVCC_FLAGS ${SM60FLAGS} ${SM70FLAGS})
	elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0) AND (${CUDA_VERSION} VERSION_LESS_EQUAL 11.0))
		message(STATUS "Configuring QUICK for SM6.0, SM7.0, and SM7.5")
	    list(APPEND CUDA_NVCC_FLAGS ${SM60FLAGS} ${SM70FLAGS} ${SM75FLAGS})
	else()
		message(FATAL_ERROR "Error: Unsupported CUDA version. ${PROJECT_NAME} requires CUDA version >= 8.0 and <= 11.0.  Please upgrade your CUDA installation or disable building with CUDA.")
	endif()
else()
	if("${QUICK_USER_ARCH}" STREQUAL "pascal")
		message(STATUS "Configuring QUICK for SM6.0")
		list(APPEND CUDA_NVCC_FLAGS ${SM60FLAGS})
	elseif("${QUICK_USER_ARCH}" STREQUAL "volta")
		message(STATUS "Configuring QUICK for SM7.0")
		list(APPEND CUDA_NVCC_FLAGS ${SM70FLAGS})
	elseif("${QUICK_USER_ARCH}" STREQUAL "turing")
		message(STATUS "Configuring QUICK for SM7.5")
		list(APPEND CUDA_NVCC_FLAGS ${SM75FLAGS})
	else()
		message(FATAL_ERROR "Invalid value for QUICK_USER_ARCH")
	endif()
endif()

# optimization level
if(OPTIMIZE)
	list(APPEND CUDA_NVCC_FLAGS -O3)
else()
	list(APPEND CUDA_NVCC_FLAGS -O0)
endif()

# debug flags
list(APPEND CUDA_NVCC_FLAGS $<$<CONFIG:Debug>:-g> $<$<CONFIG:Debug>:-G> $<$<CONFIG:Debug>:-DDEBUG>)

# extra CUDA flags
list(APPEND CUDA_NVCC_FLAGS -Xptxas=-v -use_fast_math)

if(SHARED AND (TARGET_LINUX OR TARGET_OSX))
	list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
endif()

# SPDF
if(NOT NOF)
	list(APPEND CUDA_NVCC_FLAGS -DCUDA_SPDF)
endif()

set(CUDA_DEVICE_CODE_FLAGS -Xptxas --disable-optimizer-constants)

add_subdirectory(src)

#--------------------------------------------------------------	
# Exported targets
install(EXPORT QUICK 
	FILE QUICKTarget.cmake
	NAMESPACE QUICK::
	DESTINATION share/cmake/QUICK)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/QUICKConfigVersion.cmake
	VERSION ${${PROJECT_NAME}_VERSION}
	COMPATIBILITY AnyNewerVersion)

#--------------------------------------------------------------	
# build report
if(NOT INSIDE_AMBER)
	print_build_report()
endif()