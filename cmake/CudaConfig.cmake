#CUDA configuration script for AMBER

# With CMake 3.7, FindCUDA.cmake crashes when crosscompiling.

if(CROSSCOMPILE)
	message(STATUS "CUDA disabled when crosscompiling.")
	set(CUDA FALSE)
else()

	option(CUDA "Build ${PROJECT_NAME} with CUDA GPU acceleration support." FALSE)
	if(CUDA)
		find_package(CUDA REQUIRED)

		set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

		#SM9.0 = GH200
		set(SM90FLAGS -gencode arch=compute_90,code=sm_90)
		#SM8.6  -- not currently used, but should be tested on Cuda 11.1
		set(SM86FLAGS -gencode arch=compute_86,code=sm_86)
		#SM8.0 = A100
		set(SM80FLAGS -gencode arch=compute_80,code=sm_80)
		#SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX
		set(SM75FLAGS -gencode arch=compute_75,code=sm_75)        
		#SM7.0 = V100 and Volta Geforce / GTX Ampere?
		set(SM70FLAGS -gencode arch=compute_70,code=sm_70)
		#SM6.2 = ???  -- not currently used anyway
		#set(SM62FLAGS -gencode arch=compute_62,code=sm_62)
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
		#SM3.7 = GK210 = K80  -- not currently used, since SM3.0 may be better
		#set(SM37FLAGS -gencode arch=compute_37,code=sm_37)
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

		elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0) AND (${CUDA_VERSION} VERSION_LESS 11.8))
			# Implement the standard compilation rather than a warp-synchronous one, which is deprecated as of CUDA 11

			message(STATUS "Configuring for SM3.5, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5 and SM8.0")
			list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations)
		elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 11.8) AND (${CUDA_VERSION} VERSION_LESS_EQUAL 12.5))
			message(STATUS "Configuring for SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5, SM8.0, SM8.6, and SM9.0")
			list(APPEND CUDA_NVCC_FLAGS ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} ${SM86FLAGS} ${SM90FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations)

		else()
			message(FATAL_ERROR "Error: Untested CUDA version. AMBER currently requires CUDA version >= 7.5 and <=  12.5.")
		endif()

		#  Check maximum GNU compiler versions wrt cuda:
		#  Note that this check is independent of the check(s) elsewhere
		#  for the cuda versions supported by Amber.
		#  PROGRAMMER WARNING:  This code is NOT trivial.  Before you
		#  modify it, read and understand it and the stackoverflow link !
		#  https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
		#  VERSION_EQUAL 10 means 10.0, so use ranges to compare major versions.
		if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND (
		       ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15
			AND CUDA_VERSION VERSION_GREATER_EQUAL 12.4
			AND CUDA_VERSION VERSION_LESS_EQUAL 12.6 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.3
			AND CUDA_VERSION VERSION_GREATER_EQUAL 12.4
			AND CUDA_VERSION VERSION_LESS_EQUAL 12.6 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.3
			AND CUDA_VERSION VERSION_GREATER_EQUAL 12.1
			AND CUDA_VERSION VERSION_LESS_EQUAL 12.3 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.2
			AND CUDA_VERSION VERSION_GREATER_EQUAL 12
			AND CUDA_VERSION VERSION_LESS_EQUAL 12 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12
			AND CUDA_VERSION VERSION_GREATER_EQUAL 11.4.1
			AND CUDA_VERSION VERSION_LESS_EQUAL 11.8 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11
			AND CUDA_VERSION VERSION_GREATER_EQUAL 11.1
			AND CUDA_VERSION VERSION_LESS_EQUAL 11.4.0 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10
			AND CUDA_VERSION VERSION_GREATER_EQUAL 11
			AND CUDA_VERSION VERSION_LESS_EQUAL 11 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9
			AND CUDA_VERSION VERSION_GREATER_EQUAL 10.1
			AND CUDA_VERSION VERSION_LESS_EQUAL 10.2 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8
			AND CUDA_VERSION VERSION_GREATER_EQUAL 9.2
			AND CUDA_VERSION VERSION_LESS_EQUAL 10.0 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7
			AND CUDA_VERSION VERSION_GREATER_EQUAL 9.0
			AND CUDA_VERSION VERSION_LESS_EQUAL 9.1 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6
			AND CUDA_VERSION VERSION_GREATER_EQUAL 8
			AND CUDA_VERSION VERSION_LESS_EQUAL 8 )
		    OR ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5
			AND CUDA_VERSION VERSION_GREATER_EQUAL 7
			AND CUDA_VERSION VERSION_LESS_EQUAL 7 )
		) )
			message(STATUS "Checking CUDA and GNU versions -- compatible")
		elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND (
		    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.2
			OR CUDA_VERSION VERSION_GREATER 12.5
		) )
			message(STATUS "Checking CUDA and GNU versions -- compatibility unknown")
			message(STATUS "    See https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version")
		elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" )
			message(STATUS "")
			message("************************************************************")
			message("Error: Incompatible CUDA and GNU versions!")
			message("  GNU version is ${CMAKE_CXX_COMPILER_VERSION}.")
			message("  See https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version")
			message("************************************************************")
			message(STATUS "")
			message(FATAL_ERROR)
		else()
			message(STATUS "Checking CUDA and compiler versions -- compatibility unknown")
		endif()

		set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

		#the same CUDA file is used for multiple targets in PMEMD, so turn this off
		set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)

		# --------------------------------------------------------------------
		# import a couple of CUDA libraries used by PMEMD and PBSA
		find_library(CUDA_cublas_LIBRARY NAMES cublas HINTS ${NVIDIA_MATH_LIBS})
		import_library(cublas "${CUDA_cublas_LIBRARY}")
		find_library(CUDA_cufft_LIBRARY NAMES cufft HINTS ${NVIDIA_MATH_LIBS})
		import_library(cufft "${CUDA_cufft_LIBRARY}")
		find_library(CUDA_cusolver_LIBRARY NAMES cusolver HINTS ${NVIDIA_MATH_LIBS})
		import_library(cusolver "${CUDA_cusolver_LIBRARY}")
		find_library(CUDA_curand_LIBRARY NAMES curand HINTS ${NVIDIA_MATH_LIBS})
		import_library(curand "${CUDA_curand_LIBRARY}")
		find_library(CUDA_cusparse_LIBRARY NAMES cusparse HINTS ${NVIDIA_MATH_LIBS})
		import_library(cusparse "${CUDA_cusparse_LIBRARY}")
		find_library(CUDA_cudadevrt_LIBRARY NAMES cudadevrt HINTS ${NVIDIA_MATH_LIBS})
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
		    DOC "Path to the CUDA Nvidia Management Library"
		    NO_DEFAULT_PATH)

		if(NOT EXISTS "${CUDA_nvidia-ml_LIBRARY}")
			message(WARNING "Cannot find the NVidia Management Library (libnvidia-ml) in your CUDA toolkit.  mdgx.cuda will not be built.")
		else()
	 		import_library(nvidia-ml "${CUDA_nvidia-ml_LIBRARY}")
	 	endif()

	endif()

	option(HIP "Build ${PROJECT_NAME} with HIP GPU acceleration support." FALSE)
	option(HIP_RDC "Build relocatable device code, also known as separate compilation mode." FALSE)
	option(HIP_WARP64 "Build for CDNA AMD GPUs (warp size 64) or RDNA (warp size 32)" TRUE)
	if(HIP)
		find_package(HipCUDA REQUIRED)
		set(CUDA ON)

		set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
		list(APPEND CUDA_NVCC_FLAGS
			-DAMBER_PLATFORM_AMD
			-fPIC
			-std=c++14
		)

		add_compile_definitions(AMBER_PLATFORM_AMD)
		if(HIP_WARP64)
			add_compile_definitions(AMBER_PLATFORM_AMD_WARP64)
		endif()

		set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

		#the same CUDA file is used for multiple targets in PMEMD, so turn this off
		set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)

		if(HIP_RDC)
			# Only hipcc can link a library compiled using RDC mode
			# -Wl,--unresolved-symbols=ignore-in-object-files is added after <LINK_FLAGS>
			# because CMAKE_SHARED_LINKER_FLAGS contains -Wl,--no-undefined, but we link
			# the whole program with all external shared libs later.
			set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${CUDA_NVCC_EXECUTABLE} -fgpu-rdc --hip-link <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -Wl,--unresolved-symbols=ignore-in-object-files -Wl,-soname,<TARGET> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
			# set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CUDA_NVCC_EXECUTABLE} -fgpu-rdc --hip-link <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -Wl,--unresolved-symbols=ignore-in-object-files <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
		endif()
		find_library(CUDA_cublas_LIBRARY NAMES cublas HINTS ${NVIDIA_MATH_LIBS})
		import_library(cublas "${CUDA_cublas_LIBRARY}")
		find_library(CUDA_cufft_LIBRARY NAMES cufft HINTS ${NVIDIA_MATH_LIBS})
		import_library(cufft "${CUDA_cufft_LIBRARY}")
		find_library(CUDA_curand_LIBRARY NAMES curand HINTS ${NVIDIA_MATH_LIBS})
		import_library(curand "${CUDA_curand_LIBRARY}")
		find_library(CUDA_cusparse_LIBRARY NAMES cusparse HINTS ${NVIDIA_MATH_LIBS})
		import_library(cusparse "${CUDA_cusparse_LIBRARY}")
		#import_library(cudadevrt "${CUDA_cudadevrt_LIBRARY}")
	endif()
endif()
