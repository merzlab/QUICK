set(QUICK_CUDA_CUDA_SOURCE gpu.cu gpu_type.cu gpu_get2e.cu)
     
set(QUICK_CUDA_FORTRAN_SOURCE gpu_write_info.f90)

set_property(SOURCE ${QUICK_CUDA_FORTRAN_SOURCE} PROPERTY COMPILE_OPTIONS ${OPT_FFLAGS})

#sadly FindCUDA doesn't support OBJECT libraries, so we have to use a static cuda_add_library
cuda_add_library(quick_cuda STATIC ${QUICK_CUDA_FORTRAN_SOURCE} ${QUICK_CUDA_CUDA_SOURCE} 
	OPTIONS ${CUDA_DEVICE_CODE_FLAGS} -DCUDA) 
target_compile_options(quick_cuda PRIVATE -DCUDA)
target_link_libraries(quick_cuda xc_gpu xc_cuda)
make_pic_if_needed(quick_cuda)

if(MPI)
	cuda_add_library(quick_cuda_mpi STATIC ${QUICK_CUDA_FORTRAN_SOURCE} ${QUICK_CUDA_CUDA_SOURCE} 
	OPTIONS ${CUDA_DEVICE_CODE_FLAGS} -DMPIV -DCUDA_MPIV) 
	target_compile_options(quick_cuda_mpi PRIVATE -DMPIV -DCUDA_MPIV)
	target_link_libraries(quick_cuda_mpi xc_gpu xc_cuda mpi_cxx)
	make_pic_if_needed(quick_cuda_mpi)

endif()

# CUBLAS fortran thunking
#------------------------------------------------------------------------------------------

set(CUBLAS_FORTRAN_SOURCE cublas/fortran_thunking.c)

add_library(cublas_fortran_obj OBJECT ${CUBLAS_FORTRAN_SOURCE})
set_property(TARGET cublas_fortran_obj PROPERTY COMPILE_OPTIONS ${OPT_CFLAGS}) 
set_property(TARGET cublas_fortran_obj PROPERTY COMPILE_DEFINITIONS CUBLAS_USE_THUNKING)
make_pic_if_needed(cublas_fortran_obj)

target_include_directories(cublas_fortran_obj PUBLIC ${CUDA_INCLUDE_DIRS})

# cusolver
#------------------------------------------------------------------------------------------

set(CUSOLVER_SOURCE cusolver/quick_cusolver.c)

add_library(cusolver_obj OBJECT ${CUSOLVER_SOURCE})
set_property(TARGET cusolver_obj PROPERTY COMPILE_OPTIONS ${OPT_CFLAGS}) 
make_pic_if_needed(cusolver_obj)

target_include_directories(cusolver_obj PUBLIC ${CUDA_INCLUDE_DIRS})
