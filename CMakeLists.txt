cmake_minimum_required(VERSION 3.10)
project(QUICK NONE)

include(cmake/QuickBuildSystemInit.cmake NO_POLICY_SCOPE)
enable_language(Fortran C CXX) 

set(filelist "" CACHE INTERNAL "")
set(cudalibxcobj "" CACHE INTERNAL "")
set(srcfolder "${CMAKE_SOURCE_DIR}/src")
set(exefolder "${CMAKE_SOURCE_DIR}/bin")
set(libfolder "${CMAKE_SOURCE_DIR}/lib")
set(configfolder "${srcfolder}/temp_config" )
set(subfolder "${srcfolder}/subs")
set(blasfolder "${srcfolder}/blas")
set(libxcfolder "${srcfolder}/libxc")
set(maple2c_device "${srcfolder}/libxc/maple2c_device")
set(octree "${srcfolder}/octree")
set(quick_modules "${srcfolder}/modules")
set(quick_subs "${srcfolder}/subs")
set(cudafolder "${srcfolder}/cuda")
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/")
set(BINDIR "${CMAKE_INSTALL_PREFIX}bin")
set(LIBDIR "${CMAKE_INSTALL_PREFIX}lib")

# makefolders here
file(MAKE_DIRECTORY ${exefolder} ${libfolder})

set(SERIAL false CACHE BOOL "default if no build type specified")
if((NOT SERIAL) AND (NOT USE_CUDA) AND (NOT MPI) AND (NOT CUDAMPI))
	message(STATUS "No build type specified.")
	message(STATUS "Proceeding with a serial build.")
	set(SERIAL TRUE)
endif()

include(MPIConfig)
include(CUDAConfig)

# libxc_cpu/libxc_gpu here
add_subdirectory(${libxcfolder})
if(USE_CUDA OR CUDAMPI)
	add_subdirectory(${maple2c_device})
endif()

# octree here
add_subdirectory(${octree})

# quick_modules here
add_subdirectory(${quick_modules})

# quick_subs here
add_subdirectory(${quick_subs})

# blas here
add_subdirectory(${blasfolder})
if(USE_CUDA OR CUDAMPI)
	add_subdirectory(${cudafolder})
endif()

add_subdirectory(${srcfolder})

# install data in basis dir
install(DIRECTORY basis DESTINATION .)

# install tests dir
install(DIRECTORY test DESTINATION .)

# install module dir
install(DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_DIR_NAME}/ DESTINATION ./include)

add_custom_target(test 
                 COMMAND "${CMAKE_SOURCE_DIR}/tools/runtest"
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
