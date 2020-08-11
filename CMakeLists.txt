cmake_minimum_required(VERSION 3.8.1)
project(QUICK NONE)

message("before init")
include(cmake/QuickBuildSystemInit.cmake NO_POLICY_SCOPE)
message("before enable_language")
enable_language(Fortran C CXX) #now ENABLED_LANGUAGES is a property but not a variable

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

# makefolders here
file(MAKE_DIRECTORY ${exefolder} ${libfolder})

# store mod files
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/mods)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})

include_directories(${libxcfolder})

#remove_definitions(-DCUDA -DMPI)
# added here: if MPI/CUDA, replace the config file!
include(MPIConfig)
include(CUDAConfig)

# libxc_cpu/libxc_gpu here
add_subdirectory(${libxcfolder})
if(NOT "${CUDA_ARCH}" STREQUAL "")
	add_subdirectory(${maple2c_device})
endif()

# octree here
add_subdirectory(${octree})

# quick_modules here
add_subdirectory(${quick_modules})

# quick_subs here
add_subdirectory(${quick_subs})

# blas here
if(NOT "${CUDA_ARCH}" STREQUAL "")
	add_subdirectory(${cudafolder})
else()
	message("go to cudafolder")
	add_subdirectory(${blasfolder})
endif()

# ${OBJ}, ${MAIN} and the two step compilation should all go to the ${srcfolder}
add_subdirectory(${srcfolder})


