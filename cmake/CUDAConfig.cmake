option(USE_CUDA "Build ${PROJECT_NAME} with CUDA GPU acceleration support." FALSE)

if(NOT ${CUDA_ARCH} STREQUAL "")
	set(USE_CUDA TRUE)
	enable_language(CUDA)	

	set(CMAKE_CUDA_FLAGS "-Xptxas=-v -m64 -use_fast_math")
	set(CUDA_DEVICE_CODE_FLAGS "-Xptxas --disable-optimizer-constants") 	

	include_directories(${CUDA_INCLUDE_DIRS})	

	#SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX
	set(SM75FLAGS "-gencode arch=compute_75,code=sm_75")        
	#SM7.0 = V100 and Volta Geforce / GTX Ampere?
	set(SM70FLAGS "-gencode arch=compute_70,code=sm_70")
	#SM6.0 = GP100 / P100 = DGX-1
	set(SM60FLAGS "-gencode arch=compute_60,code=sm_60")
	
	if(${CUDA_ARCH} STREQUAL "pascal")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM60FLAGS}")
	elseif(${CUDA_ARCH} STREQUAL "volta")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM70FLAGS}")
	elseif(${CUDA_ARCH} STREQUAL "turing")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM75FLAGS}")
	elseif(${CUDA_ARCH} STREQUAL "")	#if ${CUDA_ARCH} was given from the command line, we need determine ${CMAKE_CUDA_FLAGS} from ${CMAKE_CUDA_COMPILER_VERSION}
		if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL 8.0)
			message(STATUS "Configuring QUICK for SM6.0")
			set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM60FLAGS}")
		elseif((${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 9.0) AND (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 10.0)) 		
			message(STATUS "Configuring QUICK for SM6.0 and SM7.0")
			set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM60FLAGS} ${SM70FLAGS}")
		elseif((${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 10.0) AND (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS_EQUAL 11.0))
			message(STATUS "Configuring QUICK for SM6.0, SM7.0, and SM7.5")
			set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SM60FLAGS} ${SM70FLAGS} ${SM75FLAGS}")		
		else()
			message(FATAL_ERROR "Error: Unsupported CUDA version. ${PROJECT_NAME} requires CUDA version >= 8.0 and <= 11.0.  Please upgrade your CUDA installation or disable building with CUDA.")
		endif()
	endif()
endif()


# optimization level
if(NOT DEBUG)
    list(APPEND CUDAFLAGS -O3)
else()
    list(APPEND CUDAFLAGS -O0)
endif()

# debug flags
list(APPEND CUDAFLAGS $<$<CONFIG:Debug>:-g> $<$<CONFIG:Debug>:-G> $<$<CONFIG:Debug>:-DDEBUG>)

if(SHARED)	#-DSHARED=TRUE FLAG  
	list(APPEND CUDAFLAGS --compiler-options -fPIC)
endif()

# SPDF
option(NOF "Disables the compilation of QUICK's time consuming f functions in the ERI code of cuda version. Not recommended for production." FALSE)
if(NOT NOF)	#-DNOF=TRUE FLAF 
    list(APPEND CUDAFLAGS -DCUDA_SPDF)
endif()

