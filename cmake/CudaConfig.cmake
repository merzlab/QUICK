#CUDA configuration script for AMBER

# With CMake 3.7, FindCUDA.cmake crashes when crosscompiling.

if(CROSSCOMPILE)
	message(STATUS "CUDA disabled when crosscompiling.")
	set(CUDA FALSE)
else()

	# first, find CUDA.
	find_package(CUDA)
	option(CUDA "Build ${PROJECT_NAME} with CUDA GPU acceleration support." FALSE)
	
	if(CUDA AND NOT CUDA_FOUND)
		message(FATAL_ERROR "You turned on CUDA, but it was not found.  Please set the CUDA_TOOLKIT_ROOT_DIR option to your CUDA install directory.")
	endif()
	
	if(CUDA)
	
		set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

		#Note at present we do not include SM3.5 or SM3.7 since they sometimes show performance
		#regressions over just using SM3.0.
        #SM8.0 = A100
        set(SM80FLAGS -gencode arch=compute_80,code=sm_80)
        #SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX
		set(SM75FLAGS -gencode arch=compute_60,code=sm_75)        
		#SM7.0 = V100 and Volta Geforce / GTX Ampere?
		set(SM70FLAGS -gencode arch=compute_60,code=sm_70)
		#SM6.2 = ??? 
		set(SM62FLAGS -gencode arch=compute_62,code=sm_62)
		#SM6.1 = GP106 = GTX-1070, GP104 = GTX-1080, GP102 = Titan-X[P]
		set(SM61FLAGS -gencode arch=compute_61,code=sm_61)
		#SM6.0 = GP100 / P100 = DGX-1
		set(SM60FLAGS -gencode arch=compute_60,code=sm_60)
		#SM5.3 = GM200 [Grid] = M60, M40?
		set(SM53FLAGS -gencode arch=compute_53,code=sm_53)
		#SM5.2 = GM200 = GTX-Titan-X, M6000 etc.
		set(SM52FLAGS -gencode arch=compute_52,code=sm_52)
		#SM5.0 = GM204 = GTX980, 970 etc
		set(SM50FLAGS -gencode arch=compute_50,code=sm_50)
		#SM3.7 = GK210 = K80
		set(SM37FLAGS -gencode arch=compute_37,code=sm_37)
		#SM3.5 = GK110 + 110B = K20, K20X, K40, GTX780, GTX-Titan, GTX-Titan-Black, GTX-Titan-Z
		set(SM35FLAGS -gencode arch=compute_35,code=sm_35)
		#SM3.0 = GK104 = K10, GTX680, 690 etc.
		set(SM30FLAGS -gencode arch=compute_30,code=sm_30)
	
		message(STATUS "CUDA version ${CUDA_VERSION} detected")
		
		if(${CUDA_VERSION} VERSION_EQUAL 7.5)
			message(STATUS "Configuring CUDA for SM3.0, SM5.0, SM5.2 and SM5.3")
			message(STATUS "BE AWARE: CUDA 7.5 does not support GTX-1080, Titan-XP, DGX-1, V100 or other Pascal/Volta based GPUs.")
		  	list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS})
		  	
		elseif(${CUDA_VERSION} VERSION_EQUAL 8.0)
			message(STATUS "Configuring CUDA for SM3.0, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1 and SM6.2")
			message(STATUS "BE AWARE: CUDA 8.0 does not support V100, GV100, Titan-V or later GPUs")
		  	list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} -Wno-deprecated-gpu-targets)
		  	
		elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0) AND (${CUDA_VERSION} VERSION_LESS 10.0)) 
			message(STATUS "Configuring for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1 and SM7.0")
			list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations)
		elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0) AND (${CUDA_VERSION} VERSION_LESS 11.0))
			message(STATUS "Configuring for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0 and SM7.5")
			list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations)	
		elseif(${CUDA_VERSION} VERSION_EQUAL 11.0)
                        # Implement the standard compilation rather than a warp-synchronous one, which is deprecated as of CUDA 11
		        set(SM70FLAGS -gencode arch=compute_70,code=sm_70)
			set(SM75FLAGS -gencode arch=compute_75,code=sm_75)
            message(STATUS "Configuring for SM3.5, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5 and SM8.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations)                                      
		else()
			message(FATAL_ERROR "Error: Unsupported CUDA version. AMBER requires CUDA version >= 7.5 and <= 11.0.
				Please upgrade your CUDA installation or disable building with CUDA.")
		endif()
						
		set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
				
		#the same CUDA file is used for multiple targets in PMEMD, so turn this off
		set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)
				
		# --------------------------------------------------------------------
		# import a couple of CUDA libraries used by amber tools

		import_library(cublas "${CUDA_cublas_LIBRARY}")
		import_library(cufft "${CUDA_cufft_LIBRARY}")
		import_library(cusolver "${CUDA_cusolver_LIBRARY}")
		import_library(curand "${CUDA_curand_LIBRARY}")
		import_library(cusparse "${CUDA_cusparse_LIBRARY}")
	 	import_library(cudadevrt "${CUDA_cudadevrt_LIBRARY}")

	 	# --------------------------------------------------------------------
		# Find the NVidia Management Library (used to detect GPUs)
		# This library lives under the "stubs" subdirectory so we have to handle it specially.

		find_library(CUDA_nvidia-ml_LIBRARY
		    NAMES nvidia-ml
		    PATHS "${CUDA_TOOLKIT_TARGET_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
		    ENV CUDA_PATH
		    ENV CUDA_LIB_PATH
		    PATH_SUFFIXES lib64/stubs lib/stubs
		    DOC "Path to the CUDA Nvidia Management Library")

		if(NOT EXISTS "${CUDA_nvidia-ml_LIBRARY}")
			message(WARNING "Cannot find the NVidia Management Library (libnvidia-ml) in your CUDA toolkit.  mdgx.cuda will not be built.") 
		else()
	 		import_library(nvidia-ml "${CUDA_nvidia-ml_LIBRARY}")
	 	endif()


	endif()
endif()
