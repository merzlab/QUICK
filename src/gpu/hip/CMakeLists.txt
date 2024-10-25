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

if (${HIP_VERSION} VERSION_LESS 6.0.0)
	set(ROCBLAS_FORTRAN_SOURCE ${HIP_TOOLKIT_ROOT_DIR}/rocblas/include/rocblas_module.f90 rocblas/rocblas_extra_module.f90 rocblas/quick_rocblas_module.f90)
else()
	set(ROCBLAS_FORTRAN_SOURCE ${HIP_TOOLKIT_ROOT_DIR}/include/rocblas/rocblas_module.f90 rocblas/rocblas_extra_module.f90 rocblas/quick_rocblas_module.f90)
endif()

add_library(rocblas_fortran_obj OBJECT ${ROCBLAS_FORTRAN_SOURCE})
set_property(TARGET rocblas_fortran_obj PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS}) 
set_property(TARGET rocblas_fortran_obj PROPERTY COMPILE_DEFINITIONS CUBLAS_USE_THUNKING)
set_property(TARGET rocblas_fortran_obj PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(rocblas_fortran_obj PUBLIC ${CUDA_INCLUDE_DIRS})
config_module_dirs(rocblas_fortran_obj quick/${QUICK_GPU_TARGET_NAME})

if(NOT MAGMA)
	# rocsolver
	#------------------------------------------------------------------------------------------
	set(ROCSOLVER_SOURCE rocsolver/quick_rocsolver_module.f90  rocsolver/rocsolver_extra_module.f90)

	add_library(rocsolver_obj OBJECT ${ROCSOLVER_SOURCE})
	set_property(TARGET rocsolver_obj PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS}) 
	set_property(TARGET rocsolver_obj PROPERTY POSITION_INDEPENDENT_CODE TRUE)

	add_dependencies(rocsolver_obj rocblas_fortran_obj)
	target_include_directories(rocsolver_obj PUBLIC ${CUDA_INCLUDE_DIRS})
	config_module_dirs(rocsolver_obj quick/${QUICK_GPU_TARGET_NAME})
endif()

# rocblas and rocsolver results in fortran modules. Copy the targets into mpi_cuda folder. 
if(MPI)
	copy_target(rocblas_fortran_obj rocblas_fortran_obj_mpi SWAP_SOURCES TO)
	config_module_dirs(rocblas_fortran_obj_mpi quick/mpi_${QUICK_GPU_TARGET_NAME})
	set_property(TARGET rocblas_fortran_obj_mpi PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS})

	if(NOT MAGMA)
		copy_target(rocsolver_obj rocsolver_obj_mpi SWAP_SOURCES TO)
		config_module_dirs(rocsolver_obj_mpi quick/mpi_${QUICK_GPU_TARGET_NAME})
		set_property(TARGET rocsolver_obj_mpi PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS})
		add_dependencies(rocsolver_obj_mpi rocblas_fortran_obj_mpi)
	endif()
endif()
