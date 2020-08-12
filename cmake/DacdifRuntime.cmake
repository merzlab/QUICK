#File used to run a test and use dacdif to check the results.

#Since CTest can only run one command per test, we run this file and invke both commands from here.

# Variables to define:
# TEST_COMMAND - command to execute
# BASH - path to bash shell
# DACDIFF_LOCATION - path to the dacdiff script
# DACDIFF_ARGS - list of arguments to pass to dacdiff
# NDIFF_LOCATION - path to ndiff.awk
# SAVE_FILES - list of "correct" output files to compare against
# OUTPUT_FILES - list of result files of the command
# STDIN_FILE - file to use as standard input for COMMAND.  Ignored if empty string.
# SOURCE_DIR - source directory to run COMMAND in and find save files in

# the file at index N of OUTPUT_FILES is compared with the file at index N of SAVE_FILES
# In other words, put the output and save files in the same order in the list

message(STATUS "Running command (note args are seperated by semicolons) \"${TEST_COMMAND}\"")


if("${STDIN_FILE}" STREQUAL "")
	execute_process(COMMAND ${TEST_COMMAND}
			RESULT_VARIABLE COMMAND_RESULT
			OUTPUT_FILE ${STDOUT_FILE}
			ERROR_FILE ${STDERR_FILE}
			WORKING_DIRECTORY ${SOURCE_DIR})
else() #set the file as stdin

	execute_process(COMMAND ${TEST_COMMAND}
			RESULT_VARIABLE COMMAND_RESULT
			OUTPUT_FILE ${STDOUT_FILE}
			ERROR_FILE ${STDERR_FILE}
			WORKING_DIRECTORY ${SOURCE_DIR}
			INPUT_FILE ${SOURCE_DIR}/${STDIN_FILE})
endif()

if(NOT ${COMMAND_RESULT} EQUAL 0)
	message(FATAL_ERROR "Command \"${TEST_COMMAND}\" failed with status ${COMMAND_RESULT}")
endif()

foreach(OUTPUT_FILE ${OUTPUT_FILES})

	#get the corresponing save file
	list(FIND OUTPUT_FILES ${OUTPUT_FILE} FILE_INDEX)
	list(GET SAVE_FILES ${FILE_INDEX} SAVE_FILE)

	set(SAVE_FILE ${SOURCE_DIR}/${SAVE_FILE})

	message(STATUS "Checking file ${OUTPUT_FILE} against ${SAVE_FILE}")

	#message("Executing: ${BASH} ${DACDIF_LOCATION} -n ${NDIFF_LOCATION} ${DACDIFF_ARGS} ${SAVE_FILE} ${OUTPUT_FILE}")
	execute_process(COMMAND ${BASH} ${DACDIF_LOCATION} -n ${NDIFF_LOCATION} ${DACDIFF_ARGS} ${SAVE_FILE} ${OUTPUT_FILE}
			OUTPUT_VARIABLE DACDIF_OUTPUT)

	message("dacdif output: ${DACDIF_OUTPUT}")

	#I don't think that dacdif can exit with a status other than 0.
	if(("${DACDIF_OUTPUT}" MATCHES ".*FAILURE.*") OR ("${DACDIF_OUTPUT}" MATCHES ".*FAILED.*"))
		message(FATAL_ERROR "dacdif reported failure when comparing output ${OUTPUT_FILE} to save ${SAVE_FILE}")
	endif()


endforeach()

