# Checks environment for magma library installation. Users can specify 
# the installation location by using MAGMA_ROOT variable. 

find_library(LIB_MAGMA NAMES magma PATHS ${MAGMA_ROOT} $ENV{MAGMA_ROOT} ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH 
	PATH_SUFFIXES lib64 lib)

if(LIB_MAGMA)
        import_library(magma "${LIB_MAGMA}")
	get_filename_component(MAGMA_LIBRARY_DIR ${LIB_MAGMA} DIRECTORY)
	find_path(MAGMA_INCLUDE_DIR NAMES magma.mod PATHS ${MAGMA_ROOT} ${MAGMA_LIBRARY_DIR}/.. PATH_SUFFIXES include)
	if(MAGMA_INCLUDE_DIR)
		include_directories(${MAGMA_INCLUDE_DIR})
	endif()
endif()

if(LIB_MAGMA AND MAGMA_INCLUDE_DIR)
	message(STATUS "Found MAGMA: ${LIB_MAGMA}")
	message(STATUS "Found magma.mod: ${MAGMA_INCLUDE_DIR}/magma.mod")
else()
	message(FATAL_ERROR "MAGMA is requested but was not found. Make sure to set MAGMA_ROOT.")
endif()
