# first, find CUDA.
option(CUDA "Build ${PROJECT_NAME} with CUDA GPU acceleration support." FALSE)

# set these variables to minimize code duplication of hip and cuda builds
set(QUICK_GPU_PLATFORM "CUDA")
set(QUICK_GPU_TARGET_NAME "cuda")
set(GPU_LD_FLAGS "") # hipcc requires special flags for linking (see below)

if(CUDA)

    find_package(CUDA REQUIRED)

    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "You turned on CUDA, but it was not found.  Please set the CUDA_TOOLKIT_ROOT_DIR option to your CUDA install directory.")
    endif()

    # cancel Amber arch flags, because quick supports different shader models
    set(CUDA_NVCC_FLAGS "")

    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

    #SM9.0 = H100, GH200 (Hopper)
    set(SM90FLAGS -gencode arch=compute_90,code=sm_90)
    #SM8.9 = L4, L40 (Ada Lovelace)
    set(SM89FLAGS -gencode arch=compute_89,code=sm_89)
    #SM8.6 = A2, A10, A16, A40 (Ampere)
    set(SM86FLAGS -gencode arch=compute_86,code=sm_86)
    #SM8.0 = A30, A100 (Ampere)
    set(SM80FLAGS -gencode arch=compute_80,code=sm_80)
    #SM7.5 = RTX20xx, RTX Titan, T4 and Quadro RTX (Turing)
    set(SM75FLAGS -gencode arch=compute_75,code=sm_75)        
    #SM7.0 = V100 and Volta Geforce / GTX Ampere? (Volta)
    set(SM70FLAGS -gencode arch=compute_70,code=sm_70)
    #SM6.1 = GP106 = GTX-1070, GP104 = GTX-1080, GP102 = Titan-X[P] (Pascal)
    set(SM61FLAGS -gencode arch=compute_61,code=sm_61)
    #SM6.0 = GP100 / P100 = DGX-1 (Pascal)
    set(SM60FLAGS -gencode arch=compute_60,code=sm_60)
    #SM5.3 = GM200 [Grid] = M60, M40 (Maxwell)
    set(SM53FLAGS -gencode arch=compute_53,code=sm_53)
    #SM5.2 = GM200 = GTX-Titan-X, M6000 etc (Maxwell)
    set(SM52FLAGS -gencode arch=compute_52,code=sm_52)
    #SM5.0 = GM204 = GTX980, 970 et (Maxwell)
    set(SM50FLAGS -gencode arch=compute_50,code=sm_50)
    #SM3.7 = GK210 = K80 -- not currently used, since SM3.0 may be better (Kepler)
    set(SM37FLAGS -gencode arch=compute_37,code=sm_37)
    #SM3.5 = GK110 + 110B = K20, K20X, K40, GTX780, GTX-Titan, GTX-Titan-Black, GTX-Titan-Z (Kepler)
    set(SM35FLAGS -gencode arch=compute_35,code=sm_35)
    #SM3.0 = GK104 = K10, GTX680, 690 etc. (Kepler)
    set(SM30FLAGS -gencode arch=compute_30,code=sm_30)

    message(STATUS "CUDA version ${CUDA_VERSION} detected")

    set(QUICK_USER_ARCH "" CACHE STRING "Specify QUICK gpu architecture. Applicable for cuda and cudampi versions only. If empty, QUICK will be compiled for several architectures based on the CUDA toolkit version.")

    # note: need -disable-optimizer-constants for sm <= 7.0

    if("${QUICK_USER_ARCH}" STREQUAL "")
        
        # build for all supported CUDA versions
        if(${CUDA_VERSION} VERSION_EQUAL 7.5)
            message(STATUS "Configuring QUICK for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2 and SM5.3")
            message(STATUS "BE AWARE: CUDA 7.5 does not support GTX-1080, Titan-XP, DGX-1, V100 or other Pascal/Volta based GPUs.")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

        elseif(${CUDA_VERSION} VERSION_EQUAL 8.0)
            message(STATUS "Configuring QUICK for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0 and SM6.1,")
            message(STATUS "BE AWARE: CUDA 8.0 does not support V100, GV100, Titan-V or later GPUs")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            
        elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0) AND (${CUDA_VERSION} VERSION_LESS 10.0)) 
            message(STATUS "Configuring QUICK for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1 and SM7.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

        elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0) AND (${CUDA_VERSION} VERSION_LESS 11.0))
            message(STATUS "Configuring QUICK for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0 and SM7.5")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

	elseif((${CUDA_VERSION} VERSION_EQUAL 11.0))
	    message(STATUS "Configuring QUICK for SM3.0, SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5 and SM8.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM30FLAGS} ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

	elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1) AND (${CUDA_VERSION} VERSION_LESS_EQUAL 11.7))
	    message(STATUS "Configuring QUICK for SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5, SM8.0 and SM8.6")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} ${SM86FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)

	elseif((${CUDA_VERSION} VERSION_EQUAL 11.8))
            message(STATUS "Configuring QUICK for SM3.5, SM3.7, SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5, SM8.0, SM8.6, SM8.9 and SM9.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS} ${SM37FLAGS} ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} ${SM86FLAGS} ${SM89FLAGS} ${SM90FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)          
	    
	elseif((${CUDA_VERSION} VERSION_GREATER_EQUAL 12.0) AND (${CUDA_VERSION} VERSION_LESS 12.5))
            message(STATUS "Configuring QUICK for SM5.0, SM5.2, SM5.3, SM6.0, SM6.1, SM7.0, SM7.5, SM8.0, SM8.6, SM8.9 and SM9.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM50FLAGS} ${SM52FLAGS} ${SM53FLAGS} ${SM60FLAGS} ${SM61FLAGS} ${SM70FLAGS} ${SM75FLAGS} ${SM80FLAGS} ${SM86FLAGS} ${SM89FLAGS} ${SM90FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)          

	else()
	    message(FATAL_ERROR "Error: Unsupported CUDA version. ${PROJECT_NAME} requires CUDA version >= 8.0 and <= 12.4.  Please upgrade your CUDA installation or disable building with CUDA.")
	endif()

    else()

        set(FOUND "FALSE")
        
        if("${QUICK_USER_ARCH}" MATCHES "kepler")
            message(STATUS "Configuring QUICK for SM3.5")
            list(APPEND CUDA_NVCC_FLAGS ${SM35FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
            set(DISABLE_OPTIMIZER_CONSTANTS TRUE)
            set(FOUND "TRUE")
        endif()
            
        if("${QUICK_USER_ARCH}" MATCHES "maxwell")
	    message(STATUS "Configuring QUICK for SM5.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM50FLAGS})
            list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
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

        if("${QUICK_USER_ARCH}" MATCHES "adalovelace")
            message(STATUS "Configuring QUICK for SM8.9")
            list(APPEND CUDA_NVCC_FLAGS ${SM89FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS FALSE)
            set(FOUND "TRUE")
        endif()

        if("${QUICK_USER_ARCH}" MATCHES "hopper")
            message(STATUS "Configuring QUICK for SM9.0")
            list(APPEND CUDA_NVCC_FLAGS ${SM90FLAGS})
            set(DISABLE_OPTIMIZER_CONSTANTS FALSE)
            set(FOUND "TRUE")
        endif()

        if (NOT ${FOUND})
            message(FATAL_ERROR "Invalid value for QUICK_USER_ARCH. Possible values are kepler, maxwell, pascal, volta, turing, ampere, adalovelace, and hopper.")
        endif()

    endif()

    #  check maximum GNU compiler versions wrt cuda:
    #  PROGRAMMER WARNING:  This code is NOT trivial.  Before you
    #  modify it, read and understand it and the stackoverflow link !
    #  https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
    #  VERSION_EQUAL 10 means 10.0, so use ranges to compare major versions.
    if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND (
            ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.3
              AND CUDA_VERSION VERSION_GREATER_EQUAL 12.4
              AND CUDA_VERSION VERSION_LESS_EQUAL 12.4 )
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
            OR CUDA_VERSION VERSION_GREATER 12.4
    ) )
        message(STATUS "Checking CUDA and GNU versions -- compatibility unknown")
        message(STATUS "    See https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version")
    elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" )
        message(STATUS "")
        message("************************************************************")
        message("Error: Incompatible CUDA and GNU versions")
        message(" ${CMAKE_CXX_COMPILER_VERSION}")
        message(" ${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
        message("See https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version")
        message("************************************************************")
        message(STATUS "")
        message(FATAL_ERROR)
    else()
        message(STATUS "Checking CUDA and compiler versions -- compatibility unknown")
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

    # profiling flags
    if(QUICK_PROFILE)
        list(APPEND CUDA_NVCC_FLAGS --generate-line-info)
    endif()

    # extra CUDA flags
    list(APPEND CUDA_NVCC_FLAGS -use_fast_math)

    if(TARGET_LINUX OR TARGET_OSX)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
    endif()

    # SPDF
    if(ENABLEF)
        list(APPEND CUDA_NVCC_FLAGS -DGPU_SPDF)
    endif()

    if(DISABLE_OPTIMIZER_CONSTANTS)
        set(CUDA_DEVICE_CODE_FLAGS -Xptxas --disable-optimizer-constants)
    endif()

    if(USE_LEGACY_ATOMICS)
        list(APPEND CUDA_NVCC_FLAGS -DUSE_LEGACY_ATOMICS)
    endif()
	
    if(NOT INSIDE_AMBER)
	# --------------------------------------------------------------------
	# import a couple of CUDA libraries used by amber tools

        import_library(cublas "${CUDA_cublas_LIBRARY}")
        import_library(cusolver "${CUDA_cusolver_LIBRARY}")
    endif()

endif()

#option(HIP "Build ${PROJECT_NAME} with HIP GPU acceleration support." FALSE)
#option(HIP_RDC "Build relocatable device code, also known as separate compilation mode." FALSE)
#option(HIP_WARP64 "Build for CDNA AMD GPUs (warp size 64) or RDNA (warp size 32)" TRUE)
if(HIP)
    set(QUICK_GPU_PLATFORM "HIP")
    set(QUICK_GPU_TARGET_NAME "hip")
    set(GPU_LD_FLAGS -fgpu-rdc --hip-link)
    set(HIP_TOOLKIT_ROOT_DIR "/opt/rocm" CACHE STRING "Location of the HIP toolkit")
    set(HIPCUDA_EMULATE_VERSION "10.1" CACHE STRING "CUDA emulate version")
    set(AMD_HIP_FLAGS "")

    set(CUDA ON)

    # optimization level
    if(OPTIMIZE)
        list(APPEND AMD_HIP_FLAGS -O2 -ffast-math)

        set(OPT_CXXFLAGS ${OPT_CXXFLAGS} -O2 -mtune=native)

    else()
        list(APPEND AMD_HIP_FLAGS -O0)

        set(OPT_CXXFLAGS ${OPT_CXXFLAGS} "-O0")

    endif()

    list(APPEND AMD_HIP_FLAGS -fPIC -std=c++14)
    #set(TARGET_ID_SUPPORT ON)

#    if(HIP_WARP64)
#        add_compile_definitions(QUICK_PLATFORM_AMD_WARP64)
#    endif()

    if( NOT "${QUICK_USER_ARCH}" STREQUAL "")
        set(FOUND "FALSE")
        if("${QUICK_USER_ARCH}" MATCHES "gfx908")
            message(STATUS "Configuring QUICK for gfx908")
            list(APPEND AMD_HIP_FLAGS -DUSE_LEGACY_ATOMICS)
            set(FOUND "TRUE")
        endif()

        if("${QUICK_USER_ARCH}" MATCHES "gfx90a")
            message(STATUS "Configuring QUICK for gfx90a")
            list(APPEND AMD_HIP_FLAGS -DAMD_ARCH_GFX90a)
            set(FOUND "TRUE")
        endif()

        if (NOT ${FOUND})
            message(FATAL_ERROR "Invalid value for QUICK_USER_ARCH. Possible values are gfx908, gfx90a.")
        endif()
    else()
        set(QUICK_USER_ARCH "gfx908")
        list(APPEND AMD_HIP_FLAGS -DUSE_LEGACY_ATOMICS)
        message(STATUS "AMD GPU architecture not specified. Code will be optimized for gfx908.")
    endif()

    find_package(HipCUDA REQUIRED)

    execute_process(
          COMMAND ${HIP_HIPCC_EXECUTABLE} --version
	  OUTPUT_VARIABLE HIPCC_VERSION_OUTPUT
	  RESULT_VARIABLE HIPCC_VERSION_RESULT)

    if(NOT HIPCC_VERSION_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to get ROCm/HIP version.")
    endif()

    string(REPLACE "\n" ";" HIPCC_VERSION_OUTPUT ${HIPCC_VERSION_OUTPUT})
    string(REGEX MATCH "rocm-([0-9]+).([0-9]+).([0-9]+)" _ "${HIPCC_VERSION_OUTPUT}")
    set(HIP_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(HIP_VERSION_MINOR ${CMAKE_MATCH_2})
    set(HIP_VERSION_PATCH ${CMAKE_MATCH_3})
    set(HIP_VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}" CACHE STRING "ROCm/HIP version (reported by hipcc).")
    mark_as_advanced(HIP_VERSION)
    message(STATUS "Detected ROCm/HIP version: ${HIP_VERSION}")

    #  check ROCm version (as reported by hipcc),
    #  as the QUICK HIP codes trigger a known scalar register fill/spill bug
    #  in several ROCm versions
    if ( HIP_VERSION VERSION_GREATER_EQUAL 5.4.3
	    AND HIP_VERSION VERSION_LESS_EQUAL 6.2.0 )
        message(STATUS "")
        message("************************************************************")
	message("Error: Incompatible ROCm/HIP version: ${HIP_VERSION}")
	message("The QUICK HIP codes trigger a known compiler scalar register ")
	message(" fill/spill bug in ROCm >= v5.4.3 and <= v6.2.0.")
        message("************************************************************")
        message(STATUS "")
        message(FATAL_ERROR)
    endif()

    if(QUICK_DEBUG_HIP_ASAN)
	set(QUICK_USER_ARCH "${QUICK_USER_ARCH}:xnack+")
	list(APPEND CUDA_NVCC_FLAGS -fsanitize=address -fsanitize-recover=address -shared-libsan -g --offload-arch=${QUICK_USER_ARCH})
    endif()

    list(APPEND CUDA_NVCC_FLAGS ${AMD_HIP_FLAGS})

    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

#    if(HIP_RDC)
#        # Only hipcc can link a library compiled using RDC mode
#        # -Wl,--unresolved-symbols=ignore-in-object-files is added after <LINK_FLAGS>
#        # because CMAKE_SHARED_LINKER_FLAGS contains -Wl,--no-undefined, but we link
#        # the whole program with all external shared libs later.
#        set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${CUDA_NVCC_EXECUTABLE} -fgpu-rdc --hip-link <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -Wl,--unresolved-symbols=ignore-in-object-files -Wl,-soname,<TARGET> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
#        # set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CUDA_NVCC_EXECUTABLE} -fgpu-rdc --hip-link <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -Wl,--unresolved-symbols=ignore-in-object-files <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
#    endif()

    import_library(cublas "${CUDA_cublas_LIBRARY}")
    import_library(cusolver "${CUDA_cusolver_LIBRARY}")

    if(MAGMA)
        find_package(Magma REQUIRED)
	
    endif()

endif()

