set(QUICK_HIP_SOURCE gpu.cu gpu_type.cu gpu_get2e.cu gpu_oei.cu gpu_lri.cu ../xc_redistribute.cpp gpu_get2e_grad_ffff.cu)
     
#sadly FindCUDA doesn't support OBJECT libraries, so we have to use a static cuda_add_library
cuda_add_library(quick_${QUICK_GPU_TARGET_NAME}_kernels STATIC ${QUICK_HIP_SOURCE} OPTIONS ${CUDA_DEVICE_CODE_FLAGS} -DGPU -D${QUICK_GPU_PLATFORM})
target_include_directories(quick_${QUICK_GPU_TARGET_NAME}_kernels PUBLIC ../iclass)
target_link_libraries(quick_${QUICK_GPU_TARGET_NAME}_kernels xc_gpu xc_${QUICK_GPU_TARGET_NAME})

if(MPI)
	cuda_add_library(quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi STATIC ${QUICK_HIP_SOURCE}
		OPTIONS ${CUDA_DEVICE_CODE_FLAGS} -DMPIV -DMPIV_GPU -D${QUICK_GPU_PLATFORM}_MPIV) 
	target_compile_definitions(quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi PRIVATE MPIV_GPU ${QUICK_GPU_PLATFORM}_MPIV)
	set_property(TARGET quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi PROPERTY POSITION_INDEPENDENT_CODE TRUE)
        target_include_directories(quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi PUBLIC ../iclass)
	target_link_libraries(quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi xc_gpu xc_${QUICK_GPU_TARGET_NAME})

	# the generator expressions in our MPI_CXX variable don't seem to work with CUDA files, so
	# we have to link it directly
	target_link_libraries(quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi xc_gpu xc_${QUICK_GPU_TARGET_NAME} MPI::MPI_CXX)
endif()

# CUBLAS fortran thunking
#------------------------------------------------------------------------------------------

include_directories(../../util)

set(CUBLAS_FORTRAN_SOURCE ${HIP_TOOLKIT_ROOT_DIR}/rocblas/include/rocblas_module.f90 rocblas/rocblas_extra_module.f90 rocblas/quick_rocblas_module.f90)

add_library(cublas_fortran_obj OBJECT ${CUBLAS_FORTRAN_SOURCE})
set_property(TARGET cublas_fortran_obj PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS}) 
set_property(TARGET cublas_fortran_obj PROPERTY COMPILE_DEFINITIONS CUBLAS_USE_THUNKING)
set_property(TARGET cublas_fortran_obj PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(cublas_fortran_obj PUBLIC ${CUDA_INCLUDE_DIRS})
config_module_dirs(cublas_fortran_obj quick/${QUICK_GPU_TARGET_NAME})
# cusolver
#------------------------------------------------------------------------------------------
set(CUSOLVER_SOURCE rocsolver/quick_rocsolver_module.f90  rocsolver/rocsolver_extra_module.f90)

add_library(cusolver_obj OBJECT ${CUSOLVER_SOURCE})
set_property(TARGET cusolver_obj PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS}) 
set_property(TARGET cusolver_obj PROPERTY POSITION_INDEPENDENT_CODE TRUE)

add_dependencies(cusolver_obj cublas_fortran_obj)
target_include_directories(cusolver_obj PUBLIC ${CUDA_INCLUDE_DIRS})
config_module_dirs(cusolver_obj quick/${QUICK_GPU_TARGET_NAME})

# rocblas and rocsolver results in fortran modules. Copy the targets into mpi_cuda folder. 
if(MPI)
copy_target(cublas_fortran_obj cublas_fortran_obj_mpi SWAP_SOURCES TO)
config_module_dirs(cublas_fortran_obj_mpi quick/mpi_${QUICK_GPU_TARGET_NAME})
set_property(TARGET cublas_fortran_obj_mpi PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS})

copy_target(cusolver_obj cusolver_obj_mpi SWAP_SOURCES TO)
config_module_dirs(cusolver_obj_mpi quick/mpi_${QUICK_GPU_TARGET_NAME})
set_property(TARGET cusolver_obj_mpi PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS})
add_dependencies(cusolver_obj_mpi cublas_fortran_obj_mpi)
endif()

