# File that handles using an installation of Amber Host Tools to (cross-)compile Amber
# This sets the RUNNABLE_<program> variables for the rest of the build system to the versions of each program that can be run on the build system.

# Must be included after MPIConfig

set(EXECUTABLES_TO_IMPORT ucpp utilMakeHelp nab2c mpinab2c rule_parse)

set(EXECUTABLES_TO_IMPORT_REQUIRED TRUE TRUE TRUE TRUE ${MPI} TRUE) # true if the corresponding executable is needed to build Amber 

if(USE_HOST_TOOLS)
	
	if(NOT EXISTS ${HOST_TOOLS_DIR}) 
		message(FATAL_ERROR "Provided Amber Host Tools directory does not exist.  Please set HOST_TOOLS_DIR to a valid host tools directory to use host tools")
	endif()
	
	
	#import executables as "host_" versions
	foreach(EXECUTABLE ${EXECUTABLES_TO_IMPORT})
	
		# we do not know the host executable suffix, so we have to do a file search to figure out the last part of the filename
		file(GLOB EXECUTABLE_PATH_POSSIBILITIES "${HOST_TOOLS_DIR}/bin/${EXECUTABLE}*")
		list(LENGTH EXECUTABLE_PATH_POSSIBILITIES NUM_POSSIBILITIES)
		
		if(${NUM_POSSIBILITIES} EQUAL 1)
			
			set(EXECUTABLE_PATH ${EXECUTABLE_PATH_POSSIBILITIES})
			import_executable(${EXECUTABLE}_host ${EXECUTABLE_PATH})
		
			#the runnable versions are the host tools versions
			set(RUNNABLE_${EXECUTABLE} ${EXECUTABLE}_host)
			set(HAVE_RUNNABLE_${EXECUTABLE} TRUE)
		
		else()
			
			# find out if we actually need this program
			list(FIND EXECUTABLES_TO_IMPORT ${EXECUTABLE} CURR_EXECUTABLE_INDEX)
			list(GET EXECUTABLES_TO_IMPORT_REQUIRED ${CURR_EXECUTABLE_INDEX} CURR_EXECUTABLE_REQUIRED)
			
			if(CURR_EXECUTABLE_REQUIRED)
				# print an error if it was not found
				if(${NUM_POSSIBILITIES} GREATER 1)
					message(FATAL_ERROR "Multiple candidates for executable ${EXECUTABLE} in directory ${HOST_TOOLS_DIR}/bin")
				else()
					message(FATAL_ERROR "Provided Amber Host Tools directory (${HOST_TOOLS_DIR}) is missing the executable ${EXECUTABLE}")
				endif()
			endif()
		endif()
	endforeach()
else()
	#the runnable versions are the built versions
	foreach(EXECUTABLE ${EXECUTABLES_TO_IMPORT})
		set(RUNNABLE_${EXECUTABLE} ${EXECUTABLE})
		set(HAVE_RUNNABLE_${EXECUTABLE} TRUE)
	endforeach()
endif()