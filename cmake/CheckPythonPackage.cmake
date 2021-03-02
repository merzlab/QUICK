# CMake functions for checking for Python packages
# Requires PYTHON_EXECUTABLE to be defined.  Call FindPythonInterp first!

# NOTE: if moving this file, be sure to also move python_packagecheck.py


# set OUTPUT_VAR to whether PACKAGENAME was found
function(check_python_package PACKAGENAME OUTPUT_VAR)

	set(NEED_TO_RUN_CHECK TRUE)
	
	if(DEFINED ${OUTPUT_VAR})
		if(${OUTPUT_VAR})
		
			# if the python interpreter changed, we need to recheck	
			if("${PY_INTERP_FOR_${OUTPUT_VAR}}" STREQUAL "${PYTHON_EXECUTABLE}")
				set(NEED_TO_RUN_CHECK FALSE)
			endif()
			
		endif()
	endif()
	
	if(NEED_TO_RUN_CHECK)
		
		#message("${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/python_packagecheck.py ${PACKAGENAME}")
		
		set(PY_INTERP_FOR_${OUTPUT_VAR} ${PYTHON_EXECUTABLE} CACHE INTERNAL "The python interpreter used to run the ${OUTPUT_VAR} check" FORCE)
	
		execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/python_packagecheck.py ${PACKAGENAME}
			RESULT_VARIABLE PACKAGECHECK_RESULT)
		
		test(HAVE_PACKAGE ${PACKAGECHECK_RESULT} EQUAL 0)
		
		if(HAVE_PACKAGE)
			message(STATUS "Checking for Python package ${PACKAGENAME} -- found")
		else()
			message(STATUS "Checking for Python package ${PACKAGENAME} -- not found")
		endif()
		 
		set(${OUTPUT_VAR} ${HAVE_PACKAGE} CACHE BOOL "Whether the Python package ${PACKAGENAME} was found" FORCE)
		mark_as_advanced(${OUTPUT_VAR})
		
	endif()
endfunction(check_python_package)

# check that PACKAGENAME can be imported, and print an error if not
function(verify_python_package PACKAGENAME)

	# we can just generate our own variable name
	string(TOUPPER "HAVE_${PACKAGENAME}" HAVE_VAR_NAME)
	
	check_python_package(${PACKAGENAME} ${HAVE_VAR_NAME})
	
	if(NOT ${HAVE_VAR_NAME})
		message(FATAL_ERROR "The required Python package ${PACKAGENAME} was not found.  Please install it.")
	endif()
endfunction(verify_python_package)
