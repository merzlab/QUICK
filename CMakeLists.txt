#cmake_minimum_required(VERSION 3.8.1)
cmake_minimum_required(VERSION 3.10)
project(QUICK NONE)

include(cmake/QuickBuildSystemInit.cmake NO_POLICY_SCOPE)
enable_language(Fortran C CXX) #now ENABLED_LANGUAGES is a property but not a variable
set(USE_CUDA FALSE CACHE BOOL "")

message("CMAKE_C_COMPILER_ID is ${CMAKE_C_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_ID is ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_Fortran_COMPILER_ID is ${CMAKE_Fortran_COMPILER_ID}")

get_property(ALL_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
set(ENABLED_LANGUAGES "")
foreach(LANG ${ALL_ENABLED_LANGUAGES})
    if("${LANG}" STREQUAL "C" OR "${LANG}" STREQUAL "CXX" OR "${LANG}" STREQUAL "Fortran")
        list(APPEND ENABLED_LANGUAGES ${LANG})
    endif()
endforeach()

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

include(MPIConfig)
include(CUDAConfig)

# libxc_cpu/libxc_gpu here
add_subdirectory(${libxcfolder})
if(USE_CUDA)
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
if(USE_CUDA)
	add_subdirectory(${cudafolder})
endif()


add_subdirectory(${srcfolder})

# install data in basis dir
install(DIRECTORY basis DESTINATION .)

# install tests dir
install(DIRECTORY test DESTINATION .)

# install tests:
install(PROGRAMS tools/runtest DESTINATION ${CMAKE_SOURCE_DIR} RENAME quick_runtest)

# install module dir
install(DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_DIR_NAME}/ DESTINATION ./include)


