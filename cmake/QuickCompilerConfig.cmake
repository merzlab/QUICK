#cmake must have the absolute path to the compiler when setting it from a script (despite an error message to the contrary), so we make this helper function
macro(set_compiler LANGUAGE COMP_NAME)
    get_filename_component(${COMP_NAME}_LOCATION ${COMP_NAME} PROGRAM)

    if(NOT EXISTS "${${COMP_NAME}_LOCATION}") #we use EXISTS here to check that we have a full path
        # the next run should still be considered the "first" so that we can keep trying to find the compilers
        unset(FIRST_RUN CACHE)

        message(FATAL_ERROR "Could not find ${LANGUAGE} compiler executable ${COMP_NAME}.  Is it installed?")
    endif()

    message(STATUS "Setting ${LANGUAGE} compiler to ${COMP_NAME}")
    set(CMAKE_${LANGUAGE}_COMPILER ${${COMP_NAME}_LOCATION})
endmacro(set_compiler)

if(${COMPILER} STREQUAL GNU)
	#set_compiler(CUDA nvcc)
	if(${MPI})
		message("set compilers for GNU/MPI")
		set_compiler(C mpicc)
        set_compiler(CXX mpicxx)
        set_compiler(Fortran mpif90)	
	else()
		set_compiler(C gcc)
    	set_compiler(CXX g++)
    	set_compiler(Fortran gfortran)
	endif()
elseif(${COMPILER} STREQUAL INTEL)
	if(${MPI})
		message("set compilers for INTEL/MPI")
		set_compiler(C mpiicc)
    	set_compiler(CXX mpiicpc)
    	set_compiler(Fortran mpiifort)
	else()
    	set_compiler(C icc)
    	set_compiler(CXX icpc)
    	set_compiler(Fortran ifort)
	endif()
endif()

