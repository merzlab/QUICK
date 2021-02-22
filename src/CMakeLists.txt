file(GLOB filelist ${CMAKE_CURRENT_SOURCE_DIR}/*.f90 ${CMAKE_CURRENT_SOURCE_DIR}/subs/*.f90
		${CMAKE_CURRENT_SOURCE_DIR}/modules/*.f90 ${CMAKE_CURRENT_SOURCE_DIR}/octree/*.cpp)

foreach(src ${filelist})
	if(${src} MATCHES "quick_api_test.f90" OR ${src} MATCHES "main.f90")
		list(REMOVE_ITEM filelist ${src})
	endif()
endforeach()

if(SERIAL)
	add_library(quicklib ${filelist}) 
	target_compile_options(quicklib PRIVATE ${OPT_FFLAGS})
	config_module_dirs(quicklib serial)
	set_target_properties(quicklib PROPERTIES OUTPUT_NAME quick
										  ARCHIVE_OUTPUT_DIRECTORY ${libfolder}/serial 
										  LIBRARY_OUTPUT_DIRECTORY ${libfolder}/serial)
	install(TARGETS quicklib EXPORT QUICK DESTINATION ${LIBDIR}/serial)

	add_executable(quick main.f90)
	target_compile_options(quick PRIVATE ${OPT_FFLAGS})
	config_module_dirs(quick serial)
	set_property(TARGET quick PROPERTY LINKER_LANGUAGE Fortran)
	#this setting is specifically for MPI
	target_link_libraries(quick quicklib blas xc)
	set_target_properties(quick PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	install(TARGETS quick EXPORT QUICK DESTINATION ${BINDIR})

	add_executable(test-api quick_api_test.f90)
	target_compile_options(test-api PRIVATE ${OPT_FFLAGS})
	config_module_dirs(test-api serial)
	set_property(TARGET test-api PROPERTY LINKER_LANGUAGE Fortran)
	#this setting is specifically for MPI
	target_link_libraries(test-api quicklib blas xc)
	set_target_properties(test-api PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	install(TARGETS test-api EXPORT QUICK DESTINATION ${BINDIR})
endif()

if(MPI)
	add_library(quicklib_mpi ${filelist})
	target_compile_options(quicklib_mpi PRIVATE ${OPT_FFLAGS})
	config_module_dirs(quicklib_mpi mpi)
	target_compile_definitions(quicklib_mpi PRIVATE MPIV)
	set_target_properties(quicklib_mpi PROPERTIES OUTPUT_NAME quick
									   ARCHIVE_OUTPUT_DIRECTORY ${libfolder}/mpi 
									   LIBRARY_OUTPUT_DIRECTORY ${libfolder}/mpi)
	install(TARGETS quicklib_mpi EXPORT QUICK DESTINATION ${LIBDIR}/mpi)


	add_executable(quick.mpi main.f90)
	target_compile_options(quick.mpi PRIVATE ${OPT_FFLAGS})
	set_property(TARGET quick.mpi PROPERTY LINKER_LANGUAGE Fortran)
	config_module_dirs(quick.mpi mpi)
	target_compile_definitions(quick.mpi PRIVATE MPIV)
	target_link_libraries(quick.mpi quicklib_mpi xc_mpi blas_mpi)
	set_target_properties(quick.mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	install(TARGETS quick.mpi EXPORT QUICK DESTINATION ${BINDIR})

	add_executable(test-api.mpi quick_api_test.f90)
	target_compile_options(test-api.mpi PRIVATE ${OPT_FFLAGS})
	set_property(TARGET test-api.mpi PROPERTY LINKER_LANGUAGE Fortran)
    config_module_dirs(test-api.mpi mpi)
	target_compile_definitions(test-api.mpi PRIVATE MPIV)
    target_link_libraries(test-api.mpi quicklib_mpi xc_mpi blas_mpi)
    set_target_properties(test-api.mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	install(TARGETS test-api.mpi EXPORT QUICK DESTINATION ${BINDIR})
endif()

if(USE_CUDA)	
	add_library(quicklib_cuda ${filelist})
	target_compile_options(quicklib_cuda PRIVATE ${OPT_FFLAGS})
	config_module_dirs(quicklib_cuda cuda)
	target_compile_definitions(quicklib_cuda PRIVATE CUDA)
	target_link_libraries(quicklib_cuda PRIVATE gpu_all_but_xc cublas cusolver)
	set_target_properties(quicklib_cuda PROPERTIES OUTPUT_NAME quick
												   ARCHIVE_OUTPUT_DIRECTORY ${libfolder}/cuda 
												   LIBRARY_OUTPUT_DIRECTORY ${libfolder}/cuda)
	install(TARGETS quicklib_cuda EXPORT QUICK DESTINATION ${LIBDIR}/cuda)

	add_executable(quick.cuda main.f90)
	target_compile_options(quick.cuda PRIVATE ${OPT_FFLAGS})
	set_property(TARGET quick.cuda PROPERTY LINKER_LANGUAGE Fortran)
	config_module_dirs(quick.cuda cuda)
	target_compile_definitions(quick.cuda PRIVATE CUDA)
	target_link_libraries(quick.cuda quicklib_cuda xc_gpu xc_cuda -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
	set_target_properties(quick.cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	install(TARGETS quick.cuda EXPORT QUICK DESTINATION ${BINDIR})
	
	add_executable(test-api.cuda quick_api_test.f90)
	target_compile_options(test-api.cuda PRIVATE ${OPT_FFLAGS})
	set_property(TARGET test-api.cuda PROPERTY LINKER_LANGUAGE Fortran)
    config_module_dirs(test-api.cuda cuda)
	target_compile_definitions(test-api.cuda PRIVATE CUDA)
    target_link_libraries(test-api.cuda quicklib_cuda xc_gpu xc_cuda -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
    set_target_properties(test-api.cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    install(TARGETS test-api.cuda EXPORT QUICK DESTINATION ${BINDIR})
endif()

if(CUDAMPI)
	add_library(quicklib_mpi_cuda ${filelist})
	target_compile_options(quicklib_mpi_cuda PRIVATE ${OPT_FFLAGS})
    config_module_dirs(quicklib_mpi_cuda mpi_cuda)
	target_compile_definitions(quicklib_mpi_cuda PRIVATE CUDA_MPIV)
	target_link_libraries(quicklib_mpi_cuda gpu_all_but_xc_mpi cublas cusolver)	
	set_target_properties(quicklib_mpi_cuda PROPERTIES OUTPUT_NAME quick
    												  ARCHIVE_OUTPUT_DIRECTORY ${libfolder}/mpi_cuda 
													   LIBRARY_OUTPUT_DIRECTORY ${libfolder}/mpi_cuda)
    install(TARGETS quicklib_mpi_cuda EXPORT QUICK DESTINATION ${LIBDIR}/mpi_cuda)

	add_executable(quick.mpi.cuda main.f90)
	target_compile_options(quick.mpi.cuda PRIVATE ${OPT_FFLAGS})
	set_property(TARGET quick.mpi.cuda PROPERTY LINKER_LANGUAGE Fortran)
    config_module_dirs(quick.mpi.cuda mpi_cuda)
	target_compile_definitions(quick.mpi.cuda PRIVATE CUDA_MPIV)
    target_link_libraries(quick.mpi.cuda quicklib_mpi_cuda xc_mpi_gpu xc_cuda -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
	set_target_properties(quick.mpi.cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    install(TARGETS quick.mpi.cuda EXPORT QUICK DESTINATION ${BINDIR})

	add_executable(test-api.mpi.cuda quick_api_test.f90)
	target_compile_options(test-api.mpi.cuda PRIVATE ${OPT_FFLAGS})
	set_property(TARGET test-api.mpi.cuda PROPERTY LINKER_LANGUAGE Fortran)
    config_module_dirs(test-api.mpi.cuda mpi_cuda)
	target_compile_definitions(test-api.mpi.cuda PRIVATE CUDA_MPIV)
    target_link_libraries(test-api.mpi.cuda quicklib_mpi_cuda xc_mpi_gpu xc_cuda -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
	set_target_properties(test-api.mpi.cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    install(TARGETS test-api.mpi.cuda EXPORT QUICK DESTINATION ${BINDIR})
endif()
