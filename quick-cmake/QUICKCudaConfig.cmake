# first, find CUDA.
find_package(CUDA)
option(CUDA "Build ${PROJECT_NAME} with CUDA GPU acceleration support." FALSE)

if(CUDA AND NOT CUDA_FOUND)
    message(FATAL_ERROR "You turned on CUDA, but it was not found.  Please set the CUDA_TOOLKIT_ROOT_DIR option to your CUDA install directory.")
endif()

if(CUDA)

    # cancel Amber arch flags, because quick supports different shader models
    set(CUDA_NVCC_FLAGS "")

    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

    #SM8.6 -- not currently used, but should be tested on Cuda 11.1
    set(SM86FLAGS -gencode arch=compute_86,code=sm_86)
    #SM8.0 = A100
    set(SM80FLAGS -gencode arch=compute_80,code=sm_80)
    #SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX
    set(SM75FLAGS -gencode arch=compute_75,code=sm_75)        
    #SM7.0 = V100 and Volta Geforce / GTX Ampere?
    set(SM70FLAGS -gencode arch=compute_70,code=sm_70)
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
    #SM3.7 = GK210 = K80 -- not currently used, since SM3.0 may be better
    set(SM37FLAGS -gencode arch=compute_37,code=sm_37)
    #SM3.5 = GK110 + 110B = K20, K20X, K40, GTX780, GTX-Titan, GTX-Titan-Black, GTX-Titan-Z
    set(SM35FLAGS -gencode arch=compute_35,code=sm_35)
    #SM3.0 = GK104 = K10, GTX680, 690 etc.
    set(SM30FLAGS -gencode arch=compute_30,code=sm_30)

    message(STATUS "CUDA version ${CUDA_VERSION} detected")

    set(QUICK_USER_ARCH "" CACHE STRING "Specify QUICK gpu architecture. Applicable for cuda and cudampi versions only. If empty, QUICK will be compiled for several architectures based on the CUDA toolkit version.")

    # note: need -disable-optimizer-constants for sm <= 7.0

    if("${QUICK_USER_ARCH}" STREQUAL "")
        
        # build for all supported CUDA versions
	if(${CUDA_VERSION} VERSION_EQUAL 8.0)
            message(STATUS "Configuring QUICK for SM3.0, SM5.0, and SM6.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM60FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            
        elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0) AND (${CUDA_VERSION} VERSION_LESS 10.0)) 
            message(STATUS "Configuring QUICK for SM3.0, SM5.0, SM6.0 and SM7.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM60FLAGS} ${SM70FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

        elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0) AND (${CUDA_VERSION} VERSION_LESS 11.0))
            message(STATUS "Configuring QUICK for SM3.0, SM5.0, SM6.0, SM7.0 and SM7.5")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM50FLAGS} ${SM60FLAGS} ${SM70FLAGS} ${SM75FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

	elseif((${CUDA_VERSION} VERSION_EQUAL 11.0))
	    message(STATUS "Configuring QUICK for SM3.5, SM5.0, SM6.0, SM7.0, SM7.5 and SM8.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM50FLAGS} ${SM60FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

	elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1) AND (${CUDA_VERSION} VERSION_LESS_EQUAL 11.6))
	    message(STATUS "Configuring QUICK for SM3.5, SM5.0, SM6.0, SM7.0, SM7.5, SM8.0 and SM8.6")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM50FLAGS} ${SM60FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} ${SM86FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            
	else()
	    message(FATAL_ERROR "Error: Unsupported CUDA version. ${PROJECT_NAME} requires CUDA version >= 8.0 and <= 11.6.  Please upgrade your CUDA installation or disable building with CUDA.")
	endif()

    else()

        set(FOUND "FALSE")
        
        if("${QUICK_USER_ARCH}" MATCHES "kepler")
            message(STATUS "Configuring QUICK for SM3.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            set(FOUND "TRUE")
        endif()
            
        if("${QUICK_USER_ARCH}" MATCHES "maxwell")
	    message(STATUS "Configuring QUICK for SM5.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM50FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            set(FOUND "TRUE")
        endif()

        if("${QUICK_USER_ARCH}" MATCHES "pascal")
            message(STATUS "Configuring QUICK for SM6.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM60FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            set(FOUND "TRUE")
        endif()
        
        if("${QUICK_USER_ARCH}" MATCHES "volta")
            message(STATUS "Configuring QUICK for SM7.0")
	    list(APPEND CUDA_NVCC_FLAGS ${SM70FLAGS})
            if((${CUDA_VERSION} VERSION_LESS 10.0))
	        set(DISABLE_OPTIMIZER_CONSTANTS FALSE)
            endif()
            set(FOUND "TRUE")
        endif()

        if("${QUICK_USER_ARCH}" MATCHES "turing")
            message(STATUS "Configuring QUICK for SM7.5")
            list(APPEND CUDA_NVCC_FLAGS ${SM75FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS FALSE)
            set(FOUND "TRUE")
        endif()

        if("${QUICK_USER_ARCH}" MATCHES "ampere")
            message(STATUS "Configuring QUICK for SM8.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM80FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS FALSE)
            set(FOUND "TRUE")
        endif()

        if (NOT ${FOUND})
            message(FATAL_ERROR "Invalid value for QUICK_USER_ARCH")
        endif()

    endif()
					
    set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
			
    #the same CUDA file is used for multiple targets in PMEMD, so turn this off
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)

    # optimization level
    if(OPTIMIZE)
        list(APPEND CUDA_NVCC_FLAGS -O2)
    else()
        list(APPEND CUDA_NVCC_FLAGS -O0)
    endif()

    # debug flags
    list(APPEND CUDA_NVCC_FLAGS $<$<CONFIG:Debug>:-g>)
    if(QUICK_VERBOSE_PTXAS)
        list(APPEND CUDA_NVCC_FLAGS -Xptxas=-v)
    endif()

    # extra CUDA flags
    list(APPEND CUDA_NVCC_FLAGS -use_fast_math)

    if(TARGET_LINUX OR TARGET_OSX)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
    endif()

    # SPDF
    if(ENABLEF)
        list(APPEND CUDA_NVCC_FLAGS -DCUDA_SPDF)
    endif()

    if(DISABLE_OPTIMIZER_CONSTANTS)
        set(CUDA_DEVICE_CODE_FLAGS -Xptxas --disable-optimizer-constants)
    endif()

	
    if(NOT INSIDE_AMBER)
	# --------------------------------------------------------------------
	# import a couple of CUDA libraries used by amber tools

        import_library(cublas "${CUDA_cublas_LIBRARY}")
        import_library(cusolver "${CUDA_cusolver_LIBRARY}")
    endif()

endif()
