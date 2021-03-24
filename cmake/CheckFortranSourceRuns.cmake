#modified ersion of CheckCSourceRuns for Fortran code

#=============================================================================
# Copyright 2006-2009 Kitware, Inc.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#=============================================================================


#compiles the Fortran 90 free format code contained in SOURCE and runs it.
#VARIABLE will be set to TRUE for success, FALSE for failure.

#Reads the CMAKE_REQUIRED_<stuff> flags just like the the regular CheckSourceRuns
macro(check_fortran_source_runs SOURCE VAR)

	if(NOT (DEFINED "${VAR}" AND "${${VAR}_EXITCODE}" EQUAL 0))
		set(MACRO_CHECK_FUNCTION_DEFINITIONS "-D${VAR} ${CMAKE_REQUIRED_FLAGS}")
		
		if(CMAKE_REQUIRED_LIBRARIES)
			set(CHECK_Fortran_SOURCE_COMPILES_ADD_LIBRARIES LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
		else()
			set(CHECK_Fortran_SOURCE_COMPILES_ADD_LIBRARIES)
		endif()
		
		set(CHECK_Fortran_INCLUDE_OPTIONS "")
		foreach(REQ_INCLUDE ${CMAKE_REQUIRED_INCLUDES})
			list(APPEND CHECK_Fortran_INCLUDE_OPTIONS "-I${REQ_INCLUDE}")
		endforeach()
		
		file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.F90" "${SOURCE}\n")

		if(NOT CMAKE_REQUIRED_QUIET)
			message(STATUS "Performing Test ${VAR}")
		endif()
		
		try_run(${VAR}_EXITCODE ${VAR}_COMPILED
			${CMAKE_BINARY_DIR}
			${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.F90
			COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS} ${CHECK_Fortran_INCLUDE_OPTIONS}
			${CHECK_Fortran_SOURCE_COMPILES_ADD_LIBRARIES}
			CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_FUNCTION_DEFINITIONS}
			-DCMAKE_SKIP_RPATH:BOOL=${CMAKE_SKIP_RPATH}
			COMPILE_OUTPUT_VARIABLE OUTPUT)
		# if it did not compile make the return value fail code of 1
		if(NOT ${VAR}_COMPILED)
			set(${VAR}_EXITCODE 1)
		endif()
		# if the return value was 0 then it worked
		if("${${VAR}_EXITCODE}" EQUAL 0)
			set(${VAR} TRUE CACHE INTERNAL "Test ${VAR} result")
			if(NOT CMAKE_REQUIRED_QUIET)
				message(STATUS "Performed Test ${VAR} - Success")
			endif()
			file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
				"Performing Fortran SOURCE FILE Test ${VAR} succeeded with the following output:\n"
				"${OUTPUT}\n"
				"Return value: ${${VAR}}\n"
				"Source file was:\n${SOURCE}\n")
		else()
			if(CMAKE_CROSSCOMPILING AND "${${VAR}_EXITCODE}" MATCHES  "FAILED_TO_RUN")
				set(${VAR} "${${VAR}_EXITCODE}")
			else()
				set(${VAR} "" CACHE INTERNAL "Test ${VAR} result")
			endif()

			if(NOT CMAKE_REQUIRED_QUIET)
				message(STATUS "Performed Test ${VAR} - Failed")
			endif()
			file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
				"Performing Fortran SOURCE FILE Test ${VAR} failed with the following output:\n"
				"${OUTPUT}\n"
				"Return value: ${${VAR}_EXITCODE}\n"
				"Source file was:\n${SOURCE}\n")

		endif()
	endif()
endmacro(check_fortran_source_runs)
