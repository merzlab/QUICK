if(NOT ${CUDA_ARCH} STREQUAL "")
	#SET(CUDA_SEPARABLE_COMPILATION ON) # for -dc; no need
	find_package(CUDA)
	if(CUDA_FOUND)
		message("find CUDA!!!")
	endif()
	
	enable_language(CUDA)	
	set(CMAKE_CUDA_FLAGS "-O2 -Xptxas=-v -m64 -use_fast_math")
	set(CMAKE_CXX_FLAGS "-DCUBLAS_USE_THUNKING -O2 -I${CUDA_INCLUDE_DIRS}")
	
	#message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
	#message("CUDA_LIBRARIES  is ${CUDA_LIBRARIES}")
	#message("CUDA_TOOLKIT_ROOT_DIR is ${CUDA_TOOLKIT_ROOT_DIR}")	#this is CUDA_HOME
	#message("CUDA_cublas_LIBRARY is ${CUDA_cublas_LIBRARY}")	
	
	include_directories(${CUDA_INCLUDE_DIRS})	

	add_definitions(-DCUDA)
	add_definitions(-DDEBUG)
		
	if(${CUDA_ARCH} STREQUAL "pascal")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
	elseif(${CUDA_ARCH} STREQUAL "volta")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
	elseif(${CUDA_ARCH} STREQUAL "turing")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
	endif()
endif()



