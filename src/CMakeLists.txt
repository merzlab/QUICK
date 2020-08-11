file(GLOB srcsrc "*.f90") # add all *.f90 files
#list(REMOVE_ITEM ${srcsrc} "${CMAKE_CURRENT_SOURCE_DIR}/quick_api_test.f90" )

foreach(src ${srcsrc})
	get_filename_component(filename ${src} NAME)
	if(NOT ${src} MATCHES "quick_api_test.f90" AND NOT ${src} MATCHES "main.f90")
	# Above gives:
	# /opt/software/binutils/2.31.1-GCCcore-8.2.0/bin/ld: /opt/software/GCCcore/8.3.0/lib/gcc/x86_64-pc-linux-gnu/8.3.0/libgcc.a(_muldi3.o): unable to initialize decompress status for section .debug_info
	# but passes tests
	#if(NOT ${src} MATCHES "quick_api_test.f90")
		set(filelist ${filelist} ${src} "" CACHE INTERNAL "")
	endif()
endforeach()

# generate quicklib
add_library(quicklib SHARED ${filelist})
set_target_properties(quicklib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})


# generate executable quick
set(quickexe "quick")
set(test-apiexe "test-api")

if(${MPI})
	set(quickexe "quick.MPI")
	set(test-apiexe "test-api.MPI")
endif()

message("CUDA_ARCH is ${CUDA_ARCH}")
if(NOT "${CUDA_ARCH}" STREQUAL "")
    set(quickexe "quick.CUDA")
	set(test-apiexe "test-api.CUDA")
endif()

# generate executable quick/quick.MPI/quick.CUDA
add_executable(${quickexe} main.f90)
link_directories(${libfolder})
if("${CUDA_ARCH}" STREQUAL "")
	message("Not CUDA")
	target_link_libraries(${quickexe} quicklib blas xc)
else()
	target_link_libraries(${quickexe} quicklib gpu_all cublas cusolver xc -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
endif()
set_target_properties(${quickexe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# generate executable test-api
add_executable(${test-apiexe} quick_api_test.f90)
link_directories(${libfolder})
if("${CUDA_ARCH}" STREQUAL "")
	target_link_libraries(${test-apiexe} quicklib blas xc)
else()
	target_link_libraries(${test-apiexe} quicklib gpu_all cublas cusolver xc -lcublas -lcusolver -lcuda -lm -lcudart -lcudadevrt -lstdc++ ${CUDA_LIBRARIES})
endif()
set_target_properties(${test-apiexe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

