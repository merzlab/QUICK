include_directories(${CUDA_INCLUDE_DIRS} ${libxcfolder})

set(allbutxc "gpu.cu;gpu_type.cu;gpu_get2e.cu;gpu_write_info.f90;xc_redistribute.cpp")
add_library(gpu_all_but_xc STATIC ${allbutxc})
set_target_properties(gpu_all_but_xc PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_definitions(gpu_all_but_xc PUBLIC CUDA)

if(CUDAMPI)
	add_library(gpu_all_but_xc_mpi STATIC ${allbutxc})
	target_compile_definitions(gpu_all_but_xc_mpi PUBLIC CUDA_MPIV)
	set_target_properties(gpu_all_but_xc_mpi PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cublas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cusolver)

